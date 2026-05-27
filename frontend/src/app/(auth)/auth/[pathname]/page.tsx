import { viewPaths } from "@better-auth-ui/core"
import { notFound } from "next/navigation"

import { Auth } from "~/components/auth"

export default async function AuthPage({
  params
}: {
  params: Promise<{
    pathname: string
  }>
}) {
  const { pathname } = await params

  if (!Object.values(viewPaths.auth).includes(pathname)) {
    notFound()
  }

  return (
    <div className="flex justify-center my-auto p-4 md:p-6">
      <Auth path={pathname} />
    </div>
  )
}
