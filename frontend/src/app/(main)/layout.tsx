import "~/styles/globals.css"
import { Toaster } from "~/components/ui/sonner"
import { Providers } from "~/components/providers"
import { type Metadata } from "next";
import { Geist, Figtree } from "next/font/google";
import { cn } from "~/lib/utils";
import { AppSidebar } from "~/components/sidebar/app-sidebar"
import { Credits } from "~/components/sidebar/credits"
import { TooltipProvider } from "~/components/ui/tooltip"
import SoundBar from "~/components/sound-bar";

const figtree = Figtree({subsets:['latin'],variable:'--font-sans'});

export const metadata: Metadata = {
  title: "SWARA",
  description: "Music streaming platform built with T3 stack",
  icons: [{ rel: "icon", url: "/favicon.ico" }],
};

const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist-sans",
});

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={cn(geist.variable, "font-sans", figtree.variable, "dark")}
    >
      <body>
        <Providers>
          <TooltipProvider>
            <Toaster richColors position="top-right" />
            <AppSidebar creditsSlot={<Credits />} bottomSlot={<SoundBar />}>
              {children}
            </AppSidebar>
          </TooltipProvider>
        </Providers>
      </body>
    </html>
  );
}
