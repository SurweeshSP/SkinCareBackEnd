const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

class UserController {
  static async getProfile(req, res) {
    try {
      const { userId } = req.auth;
      const user = await prisma.user.findUnique({
        where: { id: userId },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          skinType: true,
          skinConcerns: true,
          allergies: true
        }
      });
      res.json({ success: true, data: user });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }

  static async updateProfile(req, res) {
    try {
      const { userId } = req.auth;
      const { skinType, conditions, allergies } = req.body;

      const updateData = { skinType };
      if (conditions) updateData.skinConcerns = JSON.stringify(conditions);
      if (allergies) updateData.allergies = JSON.stringify(allergies);

      const user = await prisma.user.update({
        where: { id: userId },
        data: updateData
      });

      res.json({ success: true, data: user });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }
}

module.exports = UserController;
