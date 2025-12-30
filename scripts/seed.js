const { PrismaClient } = require('@prisma/client');
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
require('dotenv').config({ path: '../.env' }); // Adjust path if needed

const prisma = new PrismaClient();
const s3 = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

async function main() {
    console.log("🌱 Starting Seed...");

    try {
        // 1. Create User
        const email = `seed_${Date.now()}@test.com`;
        const user = await prisma.user.upsert({
            where: { email },
            update: {},
            create: {
                email,
                password: 'hashed_dummy_password', // In real app, hash this!
                firstName: 'Seed',
                lastName: 'User',
                skinType: 'DRY'
            }
        });
        console.log(`✅ User created: ${user.email}`);

        // 2. Upload Dummy Image to S3
        const imageKey = `seed_image_${Date.now()}.txt`;
        try {
            await s3.send(new PutObjectCommand({
                Bucket: process.env.AWS_BUCKET_NAME,
                Key: imageKey,
                Body: "This is a dummy image file for connectivity testing."
            }));
            console.log(`✅ S3 Upload Check Passed: ${imageKey}`);
        } catch (e) {
            console.error(`❌ S3 Upload Failed: ${e.message}`);
            // Don't exit, we can still test DB
        }

        // 3. Create Analysis Record
        const analysis = await prisma.analysis.create({
            data: {
                userId: user.id,
                imageUrl: imageKey,
                status: 'COMPLETED',
                results: JSON.stringify({ summary: "Skin looks hydrated!" }),
                confidence: 0.95
            }
        });
        console.log(`✅ Analysis record created ID: ${analysis.id}`);

        console.log("\nDownload 'DB Browser for SQLite' to view 'prisma/dev.db' directly.");

    } catch (e) {
        console.error("❌ Seed Failed:", e);
    } finally {
        await prisma.$disconnect();
    }
}

main();
