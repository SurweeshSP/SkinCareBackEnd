const express = require('express');
const AnalysisController = require('../controllers/analysisController');
const { authenticateToken } = require('../middleware/auth');
const { validate, schemas } = require('../middleware/validation');
const rateLimit = require('express-rate-limit');

const router = express.Router();

// Rate limit: 10 analyses per hour
const limiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: 10,
  message: 'Too many requests'
});

// Mock Upload Route (Public, simulates S3 PUT)
router.put('/mock-upload', (req, res) => {
  console.log("📁 Mock Upload Received");
  res.status(200).send("Mock upload successful");
});

router.use(authenticateToken);

router.post('/create', limiter, validate(schemas.createAnalysis), AnalysisController.create);
router.get('/history', AnalysisController.getHistory);
router.post('/upload/presigned-url', AnalysisController.getPresignedUrl);
// Proxy Upload Route (Bypasses CORS issues by uploading via Backend)
router.put('/upload-proxy', async (req, res) => {
  try {
    const { key } = req.query;
    const contentType = req.headers['content-type'];

    console.log(`🚀 Proxy Uploading: ${key} (${contentType})`);

    // Buffer the stream to handle Content-Length for S3
    const chunks = [];
    for await (const chunk of req) {
      chunks.push(chunk);
    }
    const buffer = Buffer.concat(chunks);

    await require('../services/s3Service').uploadFile(key, buffer, contentType);

    res.json({ success: true, message: 'Proxy upload successful' });
  } catch (error) {
    console.error("Proxy Upload Error:", error);
    res.status(500).json({ error: error.message });
  }
});

router.get('/:id', AnalysisController.getById);
router.post('/:id/retry', AnalysisController.retry);

module.exports = router;
