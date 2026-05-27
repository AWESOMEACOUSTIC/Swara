import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { db } from "~/server/db";
import { env } from "~/env";
import { polar, checkout, portal, usage, webhooks } from "@polar-sh/better-auth";
import { Polar } from "@polar-sh/sdk";


const polarClient = new Polar({
  accessToken: env.POLAR_ACCESS_TOKEN,
  server: 'sandbox'
});

export const auth = betterAuth({
  database: prismaAdapter(db, {
    provider: "postgresql", // or "postgresql", "mysql", etc. depending on your database
  }),
  emailAndPassword: {
    enabled: true,
  },
  plugins: [
    polar({
      client: polarClient,
      createCustomerOnSignUp: true,
      use: [
        checkout({
          products: [
            {
              productId: "21d76a65-e8f0-4a91-8c22-e28abd996e5f",
              slug: "small",
            },
            {
              productId: "c32bc265-2c85-46ff-9b4c-8983cd31a144",
              slug: "medium",
            },
            {
              productId: "4430936c-c278-4276-b8e1-7d51823af521",
              slug: "large",
            },
          ],
          successUrl: "/",
          authenticatedUsersOnly: true,
        }),
        portal(),
        webhooks({
          secret: env.POLAR_WEBHOOK_SECRET,
          onOrderPaid: async (order) => {
            const externalCustomerId = order.data.customer.externalId;

            if (!externalCustomerId) {
              console.error("No external customer ID found.");
              throw new Error("No external customer id found.");
            }

            const productId = order.data.productId;

            let creditsToAdd = 0;

            switch (productId) {
              case "21d76a65-e8f0-4a91-8c22-e28abd996e5f":
                creditsToAdd = 60;
                break;
              case "c32bc265-2c85-46ff-9b4c-8983cd31a144":
                creditsToAdd = 120;
                break;
              case "4430936c-c278-4276-b8e1-7d51823af521":
                creditsToAdd = 240;
                break;
            }

            await db.user.update({
              where: { id: externalCustomerId },
              data: {
                credits: {
                  increment: creditsToAdd,
                },
              },
            });
          },
        }),
      ],
    }),
  ],
});
