"use client"

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

type AppSidebarProps = {
	children: ReactNode
}

const NAV_ITEMS = [
	{ href: "/", label: "Home" },
	{ href: "/create", label: "Create" },
	{ href: "/payments", label: "Payments" },
	{ href: "/generate", label: "Generate" },
    { href: "/settings/account", label: "Account" },
]

const formatSegmentLabel = (segment: string) =>
	segment
		.replace(/[-_]/g, " ")
		.replace(/\b\w/g, (char) => char.toUpperCase())

export function AppSidebar({ children }: AppSidebarProps) {
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
								{NAV_ITEMS.map((item) => (
									<SidebarMenuItem key={item.href}>
										<SidebarMenuButton asChild tooltip={item.label}>
											<Link href={item.href}>{item.label}</Link>
										</SidebarMenuButton>
									</SidebarMenuItem>
								))}
							</SidebarMenu>
						</SidebarGroupContent>
					</SidebarGroup>
				</SidebarContent>
				<SidebarRail />
			</Sidebar>
			<SidebarInset>
				<header className="flex h-12 items-center gap-2 border-b px-4">
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
				{children}
			</SidebarInset>
		</SidebarProvider>
	)
}
