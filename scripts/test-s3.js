const { S3Client, PutObjectCommand, ListObjectsV2Command } = require('@aws-sdk/client-s3');
require('dotenv').config({ path: '../.env' });

console.log("🌊 Starting S3 Verification...");
console.log(`   Detailed Config Check:`);
console.log(`   - Region: ${process.env.AWS_REGION}`);
console.log(`   - Bucket: ${process.env.AWS_BUCKET_NAME}`);
console.log(`   - Access Key: ${process.env.AWS_ACCESS_KEY_ID ? '******' + process.env.AWS_ACCESS_KEY_ID.slice(-4) : 'MISSING'}`);

const s3 = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

async function testS3() {
    try {
        // 1. List Files (Checks Read Permissions)
        console.log("\n1️⃣  Checking Connection (List Files)...");
        const list = await s3.send(new ListObjectsV2Command({ Bucket: process.env.AWS_BUCKET_NAME, MaxKeys: 1 }));
        console.log("   ✅ Connection Successful!");

        // 2. Upload Test File (Checks Write Permissions)
        console.log("\n2️⃣  Testing Upload...");
        const key = `test_upload_${Date.now()}.txt`;
        await s3.send(new PutObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: key,
            Body: "S3 Connectivity verified!"
        }));
        console.log(`   ✅ Upload Successful! File: ${key}`);

        console.log("\n✨ S3 Configuration is Valid!");

    } catch (error) {
        console.error("\n❌ S3 Verification Failed!");
        console.error("   Error:", error.message);
        console.error("   Solution: Check your .env file for valid AWS credentials.");
    }
}

testS3();
