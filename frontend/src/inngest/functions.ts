import { inngest } from "./client";


export const generateSong = inngest.createFunction(
  {
    id: "generate-song",
    triggers: [{ event: "generate-song-event" }]
  },
  async ({ step }) => {
    await step.sleep("wait-a-moment", "1s")
    return { message: "Song generated successfully!" }
  }
)