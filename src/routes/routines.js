const express = require('express');
const RoutineController = require('../controllers/routineController');
const { authenticateToken } = require('../middleware/auth');
const { validate, schemas } = require('../middleware/validation');

const router = express.Router();
router.use(authenticateToken);

router.post('/', validate(schemas.createRoutine), RoutineController.create);
router.get('/', RoutineController.getAll);

module.exports = router;
