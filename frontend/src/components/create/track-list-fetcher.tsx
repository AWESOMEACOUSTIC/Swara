import { headers } from "next/headers";
import { redirect } from "next/navigation";
import { getPresignedUrl } from "~/actions/generation";
import { auth } from "~/lib/auth";
import { db } from "~/server/db";
import { TrackList, type Track } from "./track-list";

export default async function TrackListFetcher() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) {
    redirect("/auth/sign-in");
  }

  const songs = await db.song.findMany({
    where: {
      userId: session.user.id,
    },
    orderBy: {
      createdAt: "desc",
    },
    select: {
      id: true,
      title: true,
      createdAt: true,
      instrumental: true,
      prompt: true,
      lyrics: true,
      described_lyrics: true,
      full_described_song: true,
      thumbnailS3Key: true,
      status: true,
      published: true,
      user: {
        select: {
          name: true,
        },
      },
    },
  });

  const tracks: Track[] = await Promise.all(
    songs.map(async (song) => ({
      id: song.id,
      title: song.title,
      createdAt: song.createdAt,
      instrumental: song.instrumental,
      prompt: song.prompt,
      lyrics: song.lyrics,
      describedLyrics: song.described_lyrics,
      described_lyrics: song.described_lyrics,
      fullDescribedSong: song.full_described_song,
      full_described_song: song.full_described_song,
      thumbnailUrl: song.thumbnailS3Key
        ? await getPresignedUrl(song.thumbnailS3Key)
        : null,
      playUrl: null,
      status: song.status,
      createdByUserName: song.user.name,
      createdByUsername: song.user.name,
      published: song.published,
    })),
  );

  return <TrackList tracks={tracks} />;
}
