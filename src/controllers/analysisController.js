const { PrismaClient } = require('@prisma/client');
const S3Service = require('../services/s3Service');
const { spawn, exec } = require('child_process');
const path = require('path');

const prisma = new PrismaClient();

/**
 * Analysis Controller - Handles all skin analysis operations
 * Features:
 * - Image upload and S3 management
 * - ML-powered skin analysis
 * - History and statistics
 * - GDPR compliance (data deletion)
 */
class AnalysisController {

  // ========================================================================
  // HELPER: Safe confidence parsing
  // ========================================================================

  static parseConfidence(value) {
    if (value === null || value === undefined) return null;
    const parsed = parseFloat(value);
    return isNaN(parsed) ? null : parsed;
  }

  // ========================================================================
  // 1. CREATE ANALYSIS - Main endpoint for new analysis
  // ========================================================================

  static async create(req, res) {
    try {
      const { userId } = req.auth;
      const { imageUrl, skinType, concerns = [] } = req.body;

      // Validate input
      if (!imageUrl) {
        return res.status(400).json({
          success: false,
          error: 'Image URL is required'
        });
      }

      const imageKey = imageUrl;

      // Create analysis record in database
      const analysis = await prisma.analysis.create({
        data: {
          userId,
          imageKey: imageKey,
          skinType: skinType || 'NORMAL',
          conditions: JSON.stringify(concerns),
          status: 'PROCESSING'
        }
      });

      // Respond immediately (async processing)
      res.status(201).json({
        success: true,
        data: {
          analysisId: analysis.id,
          status: analysis.status,
          message: 'Analysis queued for AI processing'
        }
      });

      // Execute Python ML Model asynchronously
      const pythonScript = path.join(__dirname, 'skinanalysisController.py');
      const bucket = process.env.AWS_BUCKET_NAME || 'skincare-ai-app';

      console.log(`🤖 Spawning ML Process for analysis ${analysis.id}`);
      console.log(`   Bucket: ${bucket}`);
      console.log(`   Key: ${imageKey}`);
      console.log(`   User: ${userId}`);

      // ✅ Use exec with larger buffer to prevent truncation
      const command = `python "${pythonScript}" --bucket "${bucket}" --key "${imageKey}" --userId "${userId}" --mock`;

      exec(command, {
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer (prevents truncation)
        timeout: 120000 // 2 minute timeout
      }, async (error, stdout, stderr) => {

        // Log stderr (Python logs)
        if (stderr) {
          console.log('[ML Service]', stderr);
        }

        // Handle execution errors
        if (error) {
          console.error(`❌ ML Process failed:`, error.message);
          try {
            await prisma.analysis.update({
              where: { id: analysis.id },
              data: { status: 'FAILED', completedAt: new Date() }
            });
          } catch (dbError) {
            console.error('Failed to update analysis status:', dbError);
          }
          return;
        }

        try {
          // Parse JSON output from Python
          const startMarker = "--- JSON OUTPUT START ---";
          const endMarker = "--- JSON OUTPUT END ---";
          let result;

          console.log(`[Output Debug] Length: ${stdout.length} bytes`);

          if (stdout.includes(startMarker) && stdout.includes(endMarker)) {
            // Extract JSON between markers
            const jsonStr = stdout
              .split(startMarker)[1]
              .split(endMarker)[0]
              .trim();

            console.log(`[JSON Debug] Extracted ${jsonStr.length} characters`);
            result = JSON.parse(jsonStr);
          } else {
            // Fallback: try to parse entire output
            console.warn('[Warning] JSON markers not found, parsing entire output');
            result = JSON.parse(stdout.trim());
          }

          console.log("✅ AI Results Parsed Successfully");
          console.log(`   Analysis ID: ${analysis.id}`);
          console.log(`   Skin Score: ${result.skinScore}`);
          console.log(`   Primary Condition: ${result.primaryCondition.condition}`);
          console.log(`   Skin Type: ${result.skinType}`);
          console.log(`   Confidence (raw): ${result.primaryCondition.confidence} (${typeof result.primaryCondition.confidence})`);

          // ✅ Convert confidence to Float properly
          const confidenceValue = AnalysisController.parseConfidence(
            result.primaryCondition.confidence
          );

          console.log(`   Confidence (parsed): ${confidenceValue} (${typeof confidenceValue})`);

          // Update analysis with results
          await prisma.analysis.update({
            where: { id: analysis.id },
            data: {
              status: 'COMPLETED',
              skinScore: result.skinScore,
              skinType: result.skinType || skinType,
              metrics: JSON.stringify(result.scoreBreakdown || result.metrics),
              conditions: JSON.stringify(result.detectedConditions || []),
              confidence: confidenceValue, // ✅ Now a proper Float
              completedAt: new Date()
            }
          });

          console.log(`✅ Analysis ${analysis.id} completed and saved to database`);

        } catch (parseError) {
          console.error("❌ Failed to parse ML output or update database");
          console.error("Parse error:", parseError.message);
          console.error("Stack:", parseError.stack);
          console.error("Output length:", stdout.length);
          console.error("First 500 chars:", stdout.substring(0, 500));
          console.error("Last 500 chars:", stdout.substring(Math.max(0, stdout.length - 500)));

          // Update status to FAILED
          try {
            await prisma.analysis.update({
              where: { id: analysis.id },
              data: {
                status: 'FAILED',
                completedAt: new Date()
              }
            });
          } catch (dbError) {
            console.error('Failed to update analysis status:', dbError);
          }
        }
      });

    } catch (error) {
      console.error('Create Analysis Error:', error);
      res.status(400).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 2. GET SINGLE ANALYSIS - Retrieve specific analysis result
  // ========================================================================

  static async getById(req, res) {
    try {
      const { id } = req.params;
      const { userId } = req.auth;

      // Fetch analysis (ensure user owns it)
      const analysis = await prisma.analysis.findFirst({
        where: {
          id,
          userId
        }
      });

      if (!analysis) {
        return res.status(404).json({
          success: false,
          error: 'Analysis not found'
        });
      }

      // Generate presigned URL for image viewing
      let imageUrl = null;
      if (analysis.imageKey) {
        try {
          imageUrl = await S3Service.getPresignedDownloadUrl(analysis.imageKey);
        } catch (err) {
          console.error('Failed to generate presigned URL:', err);
          // Continue without image URL (non-critical)
        }
      }

      // Parse JSON strings back to objects
      const parsedAnalysis = {
        ...analysis,
        imageUrl,
        metrics: analysis.metrics ? JSON.parse(analysis.metrics) : null,
        conditions: analysis.conditions ? JSON.parse(analysis.conditions) : [],
        confidence: analysis.confidence // Already a Float in DB
      };

      res.json({
        success: true,
        data: parsedAnalysis
      });

    } catch (error) {
      console.error('Get Analysis Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 3. GET ANALYSIS HISTORY - Paginated list of user's analyses
  // ========================================================================

  static async getHistory(req, res) {
    try {
      const { userId } = req.auth;
      const page = parseInt(req.query.page) || 1;
      const limit = Math.min(parseInt(req.query.limit) || 10, 50); // Max 50 per page
      const skip = (page - 1) * limit;

      // Fetch analyses and count in parallel
      const [analyses, total] = await Promise.all([
        prisma.analysis.findMany({
          where: { userId },
          orderBy: { createdAt: 'desc' },
          take: limit,
          skip,
          select: {
            id: true,
            imageKey: true,
            skinScore: true,
            status: true,
            confidence: true,
            createdAt: true,
            completedAt: true,
            metrics: true,
            conditions: true,
            skinType: true
          }
        }),
        prisma.analysis.count({ where: { userId } })
      ]);

      // Parse JSON fields and add image URLs
      const parsedAnalyses = await Promise.all(
        analyses.map(async (analysis) => {
          let imageUrl = null;

          // Generate presigned URL for each image
          if (analysis.imageKey) {
            try {
              imageUrl = await S3Service.getPresignedDownloadUrl(analysis.imageKey);
            } catch (err) {
              console.error(`Failed to generate URL for ${analysis.imageKey}:`, err);
            }
          }

          return {
            ...analysis,
            imageUrl,
            metrics: analysis.metrics ? JSON.parse(analysis.metrics) : null,
            conditions: analysis.conditions ? JSON.parse(analysis.conditions) : [],
            confidence: analysis.confidence // Already a Float
          };
        })
      );

      res.json({
        success: true,
        data: {
          analyses: parsedAnalyses,
          pagination: {
            page,
            limit,
            total,
            pages: Math.ceil(total / limit),
            hasNext: page * limit < total,
            hasPrev: page > 1
          }
        }
      });

    } catch (error) {
      console.error('History Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 4. GENERATE PRESIGNED URL - For client-side direct S3 upload
  // ========================================================================

  static async getPresignedUrl(req, res) {
    try {
      const { userId } = req.auth;
      const { filename = 'skin.jpg', contentType = 'image/jpeg' } = req.body;

      // Generate unique key for S3
      const timestamp = Date.now();
      const key = `analyses/${userId}/${timestamp}-${filename}`;

      // Get presigned upload URL from S3 service
      const url = await S3Service.getPresignedUploadUrl(key, contentType);

      res.json({
        success: true,
        data: {
          url,          // Presigned URL for upload
          imageKey: key, // Key to use when creating analysis
          expiresIn: 300 // Expires in 5 minutes
        }
      });

    } catch (error) {
      console.error('Presigned URL Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 5. DELETE ANALYSIS - GDPR compliance (right to be forgotten)
  // ========================================================================

  static async delete(req, res) {
    try {
      const { id } = req.params;
      const { userId } = req.auth;

      // Find analysis (ensure user owns it)
      const analysis = await prisma.analysis.findFirst({
        where: { id, userId }
      });

      if (!analysis) {
        return res.status(404).json({
          success: false,
          error: 'Analysis not found'
        });
      }

      // Delete image from S3
      if (analysis.imageKey) {
        try {
          await S3Service.deleteFile(analysis.imageKey);
          console.log(`✅ Deleted S3 file: ${analysis.imageKey}`);
        } catch (err) {
          console.error(`⚠️ Failed to delete S3 file: ${analysis.imageKey}`, err);
          // Continue with DB deletion even if S3 delete fails
        }
      }

      // Delete from database
      await prisma.analysis.delete({
        where: { id }
      });

      console.log(`✅ Deleted analysis ${id} for user ${userId}`);

      res.json({
        success: true,
        message: 'Analysis deleted successfully'
      });

    } catch (error) {
      console.error('Delete Analysis Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 6. GET STATS - Dashboard statistics
  // ========================================================================

  static async getStats(req, res) {
    try {
      const { userId } = req.auth;

      // Fetch all stats in parallel for performance
      const [stats, avgScore, total, recent] = await Promise.all([
        // Status breakdown
        prisma.analysis.groupBy({
          by: ['status'],
          where: { userId },
          _count: { id: true }
        }),

        // Average skin score (completed analyses only)
        prisma.analysis.aggregate({
          where: {
            userId,
            status: 'COMPLETED'
          },
          _avg: { skinScore: true }
        }),

        // Total count
        prisma.analysis.count({
          where: { userId }
        }),

        // Recent analyses (last 5)
        prisma.analysis.findMany({
          where: { userId },
          orderBy: { createdAt: 'desc' },
          take: 5,
          select: {
            id: true,
            skinScore: true,
            status: true,
            skinType: true,
            createdAt: true,
            completedAt: true
          }
        })
      ]);

      res.json({
        success: true,
        data: {
          totalAnalyses: total,
          avgSkinScore: Math.round(avgScore._avg.skinScore || 0),
          statusBreakdown: stats.reduce((acc, stat) => {
            acc[stat.status] = stat._count.id;
            return acc;
          }, {}),
          recent
        }
      });

    } catch (error) {
      console.error('Stats Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }

  // ========================================================================
  // 7. RETRY FAILED ANALYSIS - Reprocess failed analysis
  // ========================================================================

  static async retry(req, res) {
    try {
      const { id } = req.params;
      const { userId } = req.auth;

      // Find failed analysis
      const analysis = await prisma.analysis.findFirst({
        where: {
          id,
          userId,
          status: 'FAILED'
        }
      });

      if (!analysis) {
        return res.status(404).json({
          success: false,
          error: 'Failed analysis not found'
        });
      }

      // Update status to PROCESSING
      await prisma.analysis.update({
        where: { id: analysis.id },
        data: {
          status: 'PROCESSING',
          completedAt: null
        }
      });

      // Trigger ML processing (same as create)
      const pythonScript = path.join(__dirname, 'skinanalysisController.py');
      const bucket = process.env.AWS_BUCKET_NAME || 'skincare-ai-app';

      console.log(`🔄 Retrying analysis ${id}`);

      const command = `python "${pythonScript}" --bucket "${bucket}" --key "${analysis.imageKey}" --userId "${userId}"`;

      exec(command, {
        maxBuffer: 10 * 1024 * 1024,
        timeout: 120000
      }, async (error, stdout, stderr) => {

        if (error) {
          console.error(`❌ Retry failed:`, error.message);
          await prisma.analysis.update({
            where: { id },
            data: { status: 'FAILED', completedAt: new Date() }
          });
          return;
        }

        try {
          const startMarker = "--- JSON OUTPUT START ---";
          const endMarker = "--- JSON OUTPUT END ---";
          const jsonStr = stdout.split(startMarker)[1].split(endMarker)[0].trim();
          const result = JSON.parse(jsonStr);

          // ✅ Parse confidence properly
          const confidenceValue = AnalysisController.parseConfidence(
            result.primaryCondition.confidence
          );

          await prisma.analysis.update({
            where: { id: analysis.id },
            data: {
              status: 'COMPLETED',
              skinScore: result.skinScore,
              skinType: result.skinType,
              metrics: JSON.stringify(result.scoreBreakdown || result.metrics),
              conditions: JSON.stringify(result.detectedConditions || []),
              confidence: confidenceValue, // ✅ Float
              completedAt: new Date()
            }
          });

          console.log(`✅ Analysis ${id} retry successful`);
        } catch (err) {
          console.error(`❌ Retry parse error:`, err.message);
          await prisma.analysis.update({
            where: { id },
            data: { status: 'FAILED', completedAt: new Date() }
          });
        }
      });

      res.json({
        success: true,
        data: {
          analysisId: id,
          status: 'PROCESSING',
          message: 'Analysis retry initiated'
        }
      });

    } catch (error) {
      console.error('Retry Analysis Error:', error);
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
}

module.exports = AnalysisController;