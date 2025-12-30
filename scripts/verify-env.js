require('dotenv').config({ path: require('path').resolve(__dirname, '../.env') });
const { S3Client, PutObjectCommand, ListBucketsCommand } = require('@aws-sdk/client-s3');

async function check() {
    console.log("🔍 Verifying AWS Configuration...");

    const key = process.env.AWS_ACCESS_KEY_ID || "";
    const secret = process.env.AWS_SECRET_ACCESS_KEY || "";
    const bucket = process.env.AWS_BUCKET_NAME || "";
    const region = process.env.AWS_REGION || "";

    if (!key || !secret) {
        console.log("❌ Missing AWS Credentials in .env");
        return;
    }

    if (key.includes("your_") || secret.includes("your_")) {
        console.log("⚠️  It looks like you are using placeholder keys from the example file.");
        console.log("   - Key: " + key);
        console.log("   - Please replace them with REAL AWS keys or remove them to use Mock Mode.");
        return;
    }

    console.log(`✅ Credentials format looks OK.`);
    console.log(`   Bucket: ${bucket}`);
    console.log(`   Region: ${region}`);

    const s3 = new S3Client({
        region,
        credentials: { accessKeyId: key, secretAccessKey: secret }
    });

    try {
        console.log("\n1️⃣  Testing Auth (List Buckets)...");
        const res = await s3.send(new ListBucketsCommand({}));
        console.log("   ✅ Connection Successful! Found buckets:");
        res.Buckets.forEach(b => console.log(`      - ${b.Name}`));

        const bucketExists = res.Buckets.find(b => b.Name === bucket);
        if (!bucketExists) {
            console.log(`\n❌ Warning: Your configured bucket '${bucket}' was not found in this account!`);
            console.log("   Please check the spelling or create the bucket.");
        } else {
            console.log(`\n   ✅ Configured bucket '${bucket}' exists.`);
        }

    } catch (e) {
        console.log("\n❌ Auth Request Failed!");
        console.log("   Error:", e.name, "-", e.message);
        if (e.name === 'InvalidSignatureException') console.log("   -> Check your Secret Key.");
        if (e.name === 'UnrecognizedClientException') console.log("   -> Check your Access Key.");
        if (e.name === 'AccessDenied') console.log("   -> Your IAM User does not have permission to list buckets.");
    }

    // Test Upload
    try {
        console.log("\n2️⃣  Testing Upload Permissions...");
        await s3.send(new PutObjectCommand({
            Bucket: bucket,
            Key: 'test-permissions.txt',
            Body: 'Just checking write access'
        }));
        console.log("   ✅ Write Access Confirmed!");
    } catch (e) {
        console.log("\n❌ Upload Failed!");
        console.log("   Error:", e.name, "-", e.message);
        if (e.name === 'AccessDenied') console.log("   -> Ensure IAM User has 's3:PutObject' permission for this bucket.");
    }
}

check();
