const express = require('express');
const UserController = require('../controllers/userController');
const { authenticateToken } = require('../middleware/auth');
const { validate, schemas } = require('../middleware/validation');

const router = express.Router();
router.use(authenticateToken);

router.get('/profile', UserController.getProfile);
router.put('/profile', validate(schemas.updateProfile), UserController.updateProfile);

module.exports = router;
