const { S3Client, PutObjectCommand, DeleteObjectCommand, GetObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

const s3Client = new S3Client({
    region: process.env.AWS_REGION,
    credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
});

class S3Service {
    /**
     * Generate a presigned URL for uploading a file
     * @param {string} key - The file path/name in S3
     * @param {string} contentType - MIME type of the file
     * @param {number} expiresIn - Expiration time in seconds (default 300)
     * @returns {Promise<string>} Presigned URL
     */
    static async getPresignedUploadUrl(key, contentType = 'image/jpeg', expiresIn = 300) {
        if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'mock') {
            console.warn("⚠️ Mocking S3 Upload URL (No AWS Credentials found)");
            return `http://localhost:3000/api/analyses/mock-upload?key=${key}`;
        }

        const command = new PutObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: key,
            ContentType: contentType
        });

        return getSignedUrl(s3Client, command, { expiresIn });
    }

    /**
     * Generate a presigned URL for downloading/viewing a file
     * @param {string} key - The file path/name in S3
     * @param {number} expiresIn - Expiration time in seconds (default 3600 = 1 hour)
     * @returns {Promise<string>} Presigned URL
     */
    static async getPresignedDownloadUrl(key, expiresIn = 3600) {
        if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'mock') {
            console.warn("⚠️ Mocking S3 Download URL (No AWS Credentials found)");
            // Return a placeholder image or mock URL
            return `http://localhost:3000/api/analyses/mock-download?key=${key}`;
        }

        const command = new GetObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: key
        });

        return getSignedUrl(s3Client, command, { expiresIn });
    }

    /**
     * Delete a file from S3
     * @param {string} key - The file path/name in S3
     * @returns {Promise<void>}
     */
    static async deleteFile(key) {
        if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'mock') {
            console.log("⚠️ Mocking S3 Delete");
            return;
        }

        const command = new DeleteObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: key
        });

        await s3Client.send(command);
    }

    /**
     * Upload a file directly (Server-side)
     * @param {string} key - The file path/name in S3
     * @param {Buffer|Stream} body - File content
     * @param {string} contentType - MIME type
     */
    static async uploadFile(key, body, contentType) {
        if (!process.env.AWS_ACCESS_KEY_ID || process.env.AWS_ACCESS_KEY_ID === 'mock') {
            console.log("⚠️ Mocking Server-Side Upload");
            return;
        }

        const command = new PutObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: key,
            Body: body,
            ContentType: contentType
        });

        await s3Client.send(command);
    }
}

module.exports = S3Service;
