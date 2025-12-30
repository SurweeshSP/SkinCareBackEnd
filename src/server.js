require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');

const authRoutes = require('./routes/auth');
const analysisRoutes = require('./routes/analysis');
const userRoutes = require('./routes/users');
const routineRoutes = require('./routes/routines');

// Swagger
const swaggerUi = require('swagger-ui-express');
const YAML = require('yaml');
const fs = require('fs');
const file = fs.readFileSync('./docs/swagger.yaml', 'utf8');
const swaggerDocument = YAML.parse(file);

const app = express();

// Security & Middleware
app.use(cors({
  origin: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));
// app.use(helmet());
app.use(compression());
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Root Handler for DX
app.get('/', (req, res) => {
  res.json({
    message: 'Skin Analysis API is running',
    health: '/api/health',
    docs: '/api-docs'
  });
});

// Health Check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    routes: ['/api/auth/register', '/api/auth/login', '/api/analyses/create']
  });
});

// Debug Route: Test S3 Connection
app.get('/api/test/s3', async (req, res) => {
  try {
    const { S3Client, ListObjectsV2Command, PutObjectCommand } = require('@aws-sdk/client-s3');
    const s3 = new S3Client({
      region: process.env.AWS_REGION,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
      }
    });

    // 1. Check Config
    const config = {
      region: process.env.AWS_REGION,
      bucket: process.env.AWS_BUCKET_NAME,
      accessKeySuffix: process.env.AWS_ACCESS_KEY_ID ? process.env.AWS_ACCESS_KEY_ID.slice(-4) : 'MISSING'
    };

    // 2. Test List Permission
    await s3.send(new ListObjectsV2Command({ Bucket: process.env.AWS_BUCKET_NAME, MaxKeys: 1 }));

    // 3. Test Write Permission
    const key = `debug_test_${Date.now()}.txt`;
    await s3.send(new PutObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: key,
      Body: "Debug upload test"
    }));

    res.json({
      success: true,
      message: 'S3 Connection Verified (Read & Write)',
      config,
      testFile: key
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'S3 Check Failed',
      message: error.message,
      code: error.code || 'UNKNOWN',
      requestId: error.$metadata ? error.$metadata.requestId : null
    });
  }
});

// API Routes
app.use('/api/auth', authRoutes);
app.use('/api/analyses', analysisRoutes);
app.use('/api/users', userRoutes);
app.use('/api/routines', routineRoutes);
app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument));

// 404 Handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found',
    available: [
      '/api/health',
      '/api/auth/register',
      '/api/auth/login',
      '/api/analyses/create',
      '/api/analyses/upload/presigned-url',
      '/api/analyses/history',
      '/api/analyses/{id}',
      '/api-docs'
    ]
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('ERROR:', err.stack);
  res.status(500).json({ error: 'Internal server error' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
  console.log(`✅ Health: http://localhost:${PORT}/api/health`);
  console.log(`🔐 Auth: http://localhost:${PORT}/api/auth/register`);
  console.log(`📊 Analysis: http://localhost:${PORT}/api/analyses/create`);
});
