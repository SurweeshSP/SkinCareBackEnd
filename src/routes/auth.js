const express = require('express');
const AuthController = require('../controllers/authController');
const { validate, schemas } = require('../middleware/validation');
const { login } = require('../middleware/auth');

const router = express.Router();

router.post('/register', validate(schemas.register), AuthController.register);
router.post('/login', validate(schemas.login), login);
router.post('/reset-password', AuthController.resetPassword);

module.exports = router;
