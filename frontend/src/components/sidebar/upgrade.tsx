"use client"

import { authClient } from "~/lib/auth-client"
import { Button } from "~/components/ui/button"

export default function Upgrade() {
    const upgrade = async () => {
        const checkout = (
            authClient as unknown as {
                checkout?: (args: { products: string[] }) => Promise<void>
            }
        ).checkout

        if (!checkout) {
            console.warn("Checkout is not available on authClient")
            return
        }

        await checkout({
            products: [
                "21d76a65-e8f0-4a91-8c22-e28abd996e5f",
                "c32bc265-2c85-46ff-9b4c-8983cd31a144",
                "4430936c-c278-4276-b8e1-7d51823af521",
            ],
        })
    }

    return (
        <Button
            variant="outline"
            size="sm"
            className="cursor-pointer text-orange-400"
            onClick={upgrade}
        >
            Upgrade
        </Button>
    )
}