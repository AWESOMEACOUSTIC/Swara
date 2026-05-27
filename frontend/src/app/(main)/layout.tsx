import "~/styles/globals.css"
import { Toaster } from "~/components/ui/sonner"
import { Providers } from "~/components/providers"
import { type Metadata } from "next";
import { Geist, Figtree } from "next/font/google";
import { cn } from "~/lib/utils";
import Link from "next/link";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
} from "~/components/ui/sidebar"
import { TooltipProvider } from "~/components/ui/tooltip"

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
    <html lang="en" className={cn(geist.variable, "font-sans", figtree.variable)}>
      <body>
        <Providers>
          <TooltipProvider>
            <Toaster richColors position="top-right" />
            <SidebarProvider>
              <Sidebar>
                <SidebarHeader>
                  <div className="flex items-center gap-2 px-2 py-1.5">
                    <div className="flex size-9 items-center justify-center rounded-lg bg-sidebar-accent text-sidebar-accent-foreground">
                      <span className="text-sm font-semibold">S</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-semibold">SWARA</span>
                      <span className="text-xs text-sidebar-foreground/70">
                        Studio
                      </span>
                    </div>
                  </div>
                </SidebarHeader>
                <SidebarSeparator />
                <SidebarContent>
                  <SidebarGroup>
                    <SidebarGroupLabel>Navigation</SidebarGroupLabel>
                    <SidebarGroupContent>
                      <SidebarMenu>
                        <SidebarMenuItem>
                          <SidebarMenuButton asChild tooltip="Dashboard">
                            <Link href="/">Dashboard</Link>
                          </SidebarMenuButton>
                        </SidebarMenuItem>
                      </SidebarMenu>
                    </SidebarGroupContent>
                  </SidebarGroup>
                </SidebarContent>
                <SidebarRail />
              </Sidebar>
              <SidebarInset>
                <header className="flex h-12 items-center gap-2 border-b px-4">
                  <SidebarTrigger />
                  <span className="text-sm font-medium">Dashboard</span>
                </header>
                {children}
              </SidebarInset>
            </SidebarProvider>
          </TooltipProvider>
        </Providers>
      </body>
    </html>
  );
}
