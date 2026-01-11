const express = require('express');
const InsightsController = require('../controllers/insightsController');
const { authenticateToken } = require('../middleware/auth');

const router = express.Router();
router.use(authenticateToken);

router.get('/', InsightsController.getInsights);

module.exports = router;
