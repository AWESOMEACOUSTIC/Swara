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
                "a209b547-608c-44e7-9178-4976a73c7135",
                "11bce5cb-bfda-4c8f-afcc-4a512e2d7361",
                "7ddf3794-111c-45ba-bd4c-36935d8ed81b",
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