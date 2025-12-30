const Joi = require('joi');

const validate = (schema) => {
  return (req, res, next) => {
    const { error } = schema.validate(req.body, { abortEarly: false });
    if (error) {
      return res.status(400).json({
        error: 'Validation failed',
        details: error.details.map(d => d.message)
      });
    }
    next();
  };
};

const schemas = {
  login: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().min(6).required()
  }),
  register: Joi.object({
    email: Joi.string().email().required(),
    password: Joi.string().min(6).required(),
    firstName: Joi.string().required(),
    lastName: Joi.string().required()
  }),
  createAnalysis: Joi.object({
    imageUrl: Joi.string().required(),
    skinType: Joi.string().valid('NORMAL', 'DRY', 'OILY', 'COMBINATION', 'SENSITIVE').optional(),
    concerns: Joi.array().items(Joi.string()).optional()
  }),
  updateProfile: Joi.object({
    skinType: Joi.string().valid('NORMAL', 'DRY', 'OILY', 'COMBINATION', 'SENSITIVE', 'Normal', 'Dry', 'Oily', 'Combination', 'Sensitive').optional(),
    conditions: Joi.array().items(Joi.string()).optional(),
    allergies: Joi.array().items(Joi.string()).optional()
  }),
  createRoutine: Joi.object({
    name: Joi.string().required(),
    steps: Joi.array().items(Joi.object({
      id: Joi.string().required(),
      name: Joi.string().required(),
      product: Joi.string().allow('').optional(),
      completed: Joi.boolean().optional()
    }).unknown(true)).required(),
    frequency: Joi.string().valid('daily', 'weekly', 'monthly').required()
  })
};

module.exports = { validate, schemas };
