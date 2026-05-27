"use server"

import {redirect} from "next/navigation";
import { headers } from "next/dist/server/request/headers"
import { auth } from "~/lib/auth";
import { db } from "~/server/db"
import { inngest } from "~/inngest/client";

export async function queueSong() {
    const session = await auth.api.getSession({
        headers: await headers()
    });

    if(!session){
        redirect("/auth/sign-in")
    }

    const song = await db.song.create({
        data: {
            userId: session.user.id,
            title: "Test song 1",
            prompt: "Hip-hop song",
            full_described_song: "A hip-hop song with a fast tempo and aggressive lyrics",
        },
    });

    await inngest.send({
        name: "generate-song-event",
        data: {
            songId: song.id,
            userId: song.userId,
        },
    })
}