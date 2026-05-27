import { db } from "~/server/db";
import { env } from "~/env";
import { inngest } from "./client";

const extractSongId = (payload?: Record<string, unknown>) => {
  const directSongId =
    (payload?.songId as string | undefined) ??
    (payload?.song_id as string | undefined);

  if (directSongId) return directSongId;

  const nestedData = payload?.data as Record<string, unknown> | undefined;
  const nestedSongId =
    (nestedData?.songId as string | undefined) ??
    (nestedData?.song_id as string | undefined);

  if (nestedSongId) return nestedSongId;

  const songObj = payload?.song as { id?: string } | undefined;
  return songObj?.id;
};

export const generateSong = inngest.createFunction(
  {
    id: "generate-song",
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
    onFailure: async ({ event }) => {
      const failureData =
        (event?.data as Record<string, unknown> | undefined) ??
        (event?.data?.event as { data?: Record<string, unknown> } | undefined)
          ?.data;

      const failureSongId = extractSongId(failureData);

      if (!failureSongId) return;

      await db.song.update({
        where: {
          id: failureSongId,
        },
        data: {
          status: "failed",
        },
      });
    },
    triggers: [{ event: "generate-song-event" }],
  },
  async ({ event, step }) => {
    const eventData = event?.data as Record<string, unknown> | undefined;
    const nestedEventData =
      (eventData?.event as { data?: Record<string, unknown> } | undefined)
        ?.data;

    const songId = extractSongId(eventData) ?? extractSongId(nestedEventData);

    if (!songId) {
      console.warn("Missing songId in event data", {
        eventData: eventData ?? null,
      });
      return;
    }

    const { userId, credits, endpoint, body } = await step.run(
      "check-credits",
      async () => {
        const song = await db.song.findUniqueOrThrow({
          where: {
            id: songId,
          },
          select: {
            user: {
              select: {
                id: true,
                credits: true,
              },
            },
            prompt: true,
            lyrics: true,
            full_described_song: true,
            described_lyrics: true,
            instrumental: true,
            guidance_scale: true,
            infer_step: true,
            audio_duration: true,
            seed: true,
          },
        });

        type RequestBody = {
          guidance_scale?: number;
          infer_step?: number;
          audio_duration?: number;
          seed?: number;
          full_described_song?: string;
          prompt?: string;
          lyrics?: string;
          described_lyrics?: string;
          instrumental?: boolean;
        };

        let endpoint = "";
        let body: RequestBody = {};

        const commomParams = {
          guidance_scale: song.guidance_scale ?? undefined,
          infer_step: song.infer_step ?? undefined,
          audio_duration: song.audio_duration ?? undefined,
          seed: song.seed ?? undefined,
          instrumental: song.instrumental ?? undefined,
        };

        // Description of a song
        if (song.full_described_song) {
          endpoint = env.GENERATE_FROM_DESCRIPTION;
          body = {
            full_described_song: song.full_described_song,
            ...commomParams,
          };
        }

        // Custom mode: Lyrics + prompt
        else if (song.lyrics && song.prompt) {
          endpoint = env.GENERATE_WITH_LYRICS;
          body = {
            lyrics: song.lyrics,
            prompt: song.prompt,
            ...commomParams,
          };
        }

        // Custom mode: Prompt + described lyrics
        else if (song.described_lyrics && song.prompt) {
          endpoint = env.GENERATE_WITH_DESCRIBED_LYRICS;
          body = {
            described_lyrics: song.described_lyrics,
            prompt: song.prompt,
            ...commomParams,
          };
        }

        return {
          userId: song.user.id,
          credits: song.user.credits,
          endpoint: endpoint,
          body: body,
        };
      },
    );

    if (credits > 0) {
      // Generate the song
      await step.run("set-status-processing", async () => {
        return await db.song.update({
          where: {
            id: songId,
          },
          data: {
            status: "processing",
          },
        });
      });

      const response = await step.fetch(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          "Content-Type": "application/json",
          "Modal-Key": env.MODAL_KEY,
          "Modal-Secret": env.MODAL_SECRET,
        },
      });

      const responseData = await step.run("update-song-result", async () => {
        const payload = response.ok
          ? ((await response.json()) as {
              s3_key: string;
              cover_image_s3_key: string;
              categories: string[];
            })
          : null;

        await db.song.update({
          where: {
            id: songId,
          },
          data: {
            s3Key: payload?.s3_key,
            thumbnailS3Key: payload?.cover_image_s3_key,
            status: response.ok ? "processed" : "failed",
          },
        });

        if (payload && payload.categories.length > 0) {
          await db.song.update({
            where: { id: songId },
            data: {
              categories: {
                connectOrCreate: payload.categories.map((categoryName) => ({
                  where: { name: categoryName },
                  create: { name: categoryName },
                })),
              },
            },
          });
        }

        return payload;
      });

      await step.run("deduct-credits", async () => {
        if (!response.ok) return;

        await db.user.update({
          where: { id: userId },
          data: {
            credits: {
              decrement: 1,
            },
          },
        });
      });

      return responseData;
    } else {
      // Set song status "not enough credits"
      await step.run("set-status-no-credits", async () => {
        return await db.song.update({
          where: {
            id: songId,
          },
          data: {
            status: "no credits",
          },
        });
      });
    }
  },
);