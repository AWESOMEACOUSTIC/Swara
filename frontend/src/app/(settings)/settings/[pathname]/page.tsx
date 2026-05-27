import { viewPaths } from "@better-auth-ui/core"
import { ensureSession } from "@better-auth-ui/react/server"
import { dehydrate, HydrationBoundary } from "@tanstack/react-query"
import { headers } from "next/headers"
import { notFound, redirect } from "next/navigation"

import { Settings } from "~/components/settings"
import { auth } from "~/lib/auth"
import { getQueryClient } from "~/lib/query-client"

export default async function SettingsPage({
  params
}: {
  params: Promise<{
    pathname: string
  }>
}) {
  const { pathname } = await params

  if (!Object.values(viewPaths.settings).includes(pathname)) {
    notFound()
  }

  const requestHeaders = await headers()
  const queryClient = getQueryClient()

  const session = await ensureSession(queryClient, auth, {
    headers: requestHeaders
  })

  if (!session) {
    redirect(
      `/auth/sign-in?redirectTo=${encodeURIComponent(
        `/settings/${pathname}`
      )}`
    )
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <div className="w-full max-w-3xl mx-auto p-4 md:p-6">
        <Settings path={pathname} />
      </div>
    </HydrationBoundary>
  )
}
