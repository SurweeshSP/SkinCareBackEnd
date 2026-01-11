const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

exports.getInsights = async (req, res) => {
    try {
        const userId = req.user.userId;

        // Fetch the latest analysis
        const latestAnalysis = await prisma.analysis.findFirst({
            where: { userId },
            orderBy: { createdAt: 'desc' },
            select: {
                skinScore: true,
                metrics: true, // JSON string
                createdAt: true,
                conditions: true // JSON string
            }
        });

        if (!latestAnalysis) {
            return res.json([]); // No insights if no analysis
        }

        const insights = [];
        const metrics = JSON.parse(latestAnalysis.metrics || '{}');
        const conditions = JSON.parse(latestAnalysis.conditions || '[]');

        // 1. Hydration Insight
        if (metrics.hydration && metrics.hydration < 50) {
            insights.push({
                id: 'hydration_alert',
                title: 'Hydration Low',
                detail: 'Your skin hydration is below optimal levels. Drink 2L of water today.'
            });
        } else if (metrics.hydration > 70) {
            insights.push({
                id: 'hydration_good',
                title: 'Great Hydration',
                detail: 'Your skin moisture barrier is looking strong!'
            });
        }

        // 2. Texture/Smoothness
        if (metrics.texture && metrics.texture < 60) {
            insights.push({
                id: 'texture_fix',
                title: 'Exfoliation Needed',
                detail: 'Texture score is lower. Consider a gentle chemical exfoliant tonight.'
            });
        }

        // 3. Condition Specific
        if (conditions.length > 0) {
            const primary = conditions[0]; // Assuming sorted by confidence or severity
            insights.push({
                id: 'condition_focus',
                title: `Manage ${capitalize(primary.condition)}`,
                detail: `We detected signs of ${primary.condition}. Check your routine for targeted treatments.`
            });
        }

        // 4. General UV Advice (Randomized or Time-based for now)
        const hour = new Date().getHours();
        if (hour >= 9 && hour <= 16) {
            insights.push({
                id: 'uv_protection',
                title: 'UV Alert',
                detail: 'UV index is moderate outside. Apply SPF 30+ before going out.'
            });
        }

        // 5. Progress Insight (if previous analysis exists)
        // (Skipping for V1 to keep it simple, can add later)

        res.json(insights);
    } catch (error) {
        console.error('Insight generation error:', error);
        res.status(500).json({ error: 'Failed to generate insights' });
    }
};

function capitalize(s) {
    if (!s) return '';
    return s.charAt(0).toUpperCase() + s.slice(1);
}
