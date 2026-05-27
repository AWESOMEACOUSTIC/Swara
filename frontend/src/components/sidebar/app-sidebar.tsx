"use client"

import { Home, Plus, Sparkles, User, CreditCard, type LucideIcon } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Fragment, type ReactNode } from "react"
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from "~/components/ui/breadcrumb"
import { Separator } from "~/components/ui/separator"
import {
    Sidebar,
    SidebarContent,
    SidebarFooter,
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
import { UserButton } from "../user-button"
import Upgrade from "./upgrade"

type AppSidebarProps = {
    children: ReactNode
    creditsSlot?: ReactNode
    bottomSlot?: ReactNode
}

type NavItem = {
    href: string
    label: string
    icon?: LucideIcon
}

const NAV_ITEMS: NavItem[] = [
    { href: "/", label: "Home", icon: Home },
    { href: "/create", label: "Generate", icon: Sparkles },
    { href: "/payments", label: "Payments", icon: CreditCard },
]

const formatSegmentLabel = (segment: string) =>
    segment
        .replace(/[-_]/g, " ")
        .replace(/\b\w/g, (char) => char.toUpperCase())

export function AppSidebar({ children, creditsSlot, bottomSlot }: AppSidebarProps) {
    const pathname = usePathname()
    const segments = pathname.split("/").filter(Boolean)
    const rootCrumb =
        NAV_ITEMS.find((item) => item.href === "/") ??
        ({ href: "/", label: "Home" } satisfies (typeof NAV_ITEMS)[number])
    const breadcrumbs = [
        rootCrumb,
        ...segments.map((segment, index) => {
            const href = `/${segments.slice(0, index + 1).join("/")}`
            const navItem = NAV_ITEMS.find((item) => item.href === href)
            return {
                href,
                label: navItem?.label ?? formatSegmentLabel(segment),
            }
        }),
    ]

    return (
        <SidebarProvider>
            <Sidebar>
                <SidebarHeader>
                    <Link href="/" className="flex items-center gap-2 px-2 py-1.5">
                        <div className="flex size-9 items-center justify-center rounded-lg bg-sidebar-accent text-sidebar-accent-foreground">
                            <span className="text-sm font-semibold">S</span>
                        </div>
                        <div className="flex flex-col">
                            <span className="text-sm font-semibold">SWARA</span>
                            <span className="text-xs text-sidebar-foreground/70">
                                Studio
                            </span>
                        </div>
                    </Link>
                </SidebarHeader>
                <SidebarSeparator />
                <SidebarContent>
                    <SidebarGroup>
                        <SidebarGroupLabel>Navigation</SidebarGroupLabel>
                        <SidebarGroupContent>
                            <SidebarMenu>
                                {NAV_ITEMS.map((item) => (
                                    <SidebarMenuItem key={item.href}>
                                        <SidebarMenuButton
                                            asChild
                                            tooltip={item.label}
                                            isActive={
                                                item.href === "/"
                                                    ? pathname === "/"
                                                    : pathname.startsWith(item.href)
                                            }
                                        >
                                            <Link href={item.href}>
                                                {item.icon ? (
                                                    <item.icon className="size-4" />
                                                ) : null}
                                                <span>{item.label}</span>
                                            </Link>
                                        </SidebarMenuButton>
                                    </SidebarMenuItem>
                                ))}
                            </SidebarMenu>
                        </SidebarGroupContent>
                    </SidebarGroup>
                </SidebarContent>
                <SidebarFooter>
                    <div className="mb-2 flex w-full items-center justify-center gap-5 text-xs">
                        {creditsSlot}
                        <Upgrade />
                    </div>
                    <UserButton
                        variant="outline"
                        links={[
                            {
                                label: "Customer Portal",
                                href: "/settings/account",
                                icon: <User />,
                            },
                        ]}
                    />
                </SidebarFooter>
                <SidebarRail />
            </Sidebar>
            <SidebarInset className="flex h-svh flex-col overflow-hidden">
                <header className="flex h-12 shrink-0 items-center gap-2 border-b px-4">
                    <SidebarTrigger />
                    <Separator
                        orientation="vertical"
                        className="mx-2 h-4"
                    />
                    <Breadcrumb>
                        <BreadcrumbList>
                            {breadcrumbs.map((crumb, index) => {
                                const isLast = index === breadcrumbs.length - 1
                                return (
                                    <Fragment key={crumb.href}>
                                        <BreadcrumbItem>
                                            {isLast ? (
                                                <BreadcrumbPage>
                                                    {crumb.label}
                                                </BreadcrumbPage>
                                            ) : (
                                                <BreadcrumbLink asChild>
                                                    <Link href={crumb.href}>{crumb.label}</Link>
                                                </BreadcrumbLink>
                                            )}
                                        </BreadcrumbItem>
                                        {!isLast && <BreadcrumbSeparator />}
                                    </Fragment>
                                )
                            })}
                        </BreadcrumbList>
                    </Breadcrumb>
                </header>
                <div className="min-h-0 flex-1 overflow-hidden">
                    {children}
                </div>
                {bottomSlot ? <div className="shrink-0">{bottomSlot}</div> : null}
            </SidebarInset>
        </SidebarProvider>
    )
}
