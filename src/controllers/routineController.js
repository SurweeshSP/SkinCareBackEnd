const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

class RoutineController {
  static async create(req, res) {
    try {
      const { userId } = req.auth;
      const { name, steps, frequency } = req.body;
      const startDate = new Date();

      // Check if user has an existing active routine
      const existing = await prisma.skincareRoutine.findFirst({
        where: { userId, isActive: true }
      });

      let routine;
      const data = {
        name,
        frequency,
        steps: steps ? JSON.stringify(steps) : '[]',
        startDate,
        userId
      };

      if (existing) {
        routine = await prisma.skincareRoutine.update({
          where: { id: existing.id },
          data: {
            name,
            frequency,
            steps: steps ? JSON.stringify(steps) : '[]'
          }
        });
      } else {
        routine = await prisma.skincareRoutine.create({
          data: { ...data, isActive: true }
        });
      }

      res.status(201).json({ success: true, data: routine });
    } catch (error) {
      console.error(error);
      res.status(400).json({ error: error.message });
    }
  }

  static async getAll(req, res) {
    try {
      const { userId } = req.auth;
      const routines = await prisma.skincareRoutine.findMany({
        where: { userId }
      });
      res.json({ success: true, data: routines });
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
  }
}

module.exports = RoutineController;
