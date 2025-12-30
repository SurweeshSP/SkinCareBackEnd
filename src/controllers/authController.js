const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

class AuthController {
  static async register(req, res) {
    try {
      const { email, password, firstName, lastName } = req.body;
      const hashedPassword = bcrypt.hashSync(password, 12);

      const user = await prisma.user.create({
        data: {
          email,
          password: hashedPassword,
          firstName,
          lastName,
          skinType: 'NORMAL'
        }
      });

      const token = jwt.sign({ userId: user.id }, process.env.JWT_SECRET, { expiresIn: '24h' });
      res.status(201).json({
        success: true,
        data: {
          token,
          user: {
            id: user.id,
            email: user.email,
            firstName: user.firstName,
            lastName: user.lastName
          }
        }
      });
    } catch (error) {
      res.status(400).json({ error: 'Registration failed' });
    }
  }

  static async resetPassword(req, res) {
    try {
      const { email, newPassword } = req.body;
      const hashedPassword = bcrypt.hashSync(newPassword, 12);

      const user = await prisma.user.update({
        where: { email },
        data: { password: hashedPassword }
      });

      res.json({ success: true, message: 'Password updated successfully' });
    } catch (error) {
      // P2025 is Prisma error for record not found
      if (error.code === 'P2025') {
        return res.status(404).json({ error: 'User not found' });
      }
      res.status(500).json({ error: 'Password reset failed' });
    }
  }
}

module.exports = AuthController;
