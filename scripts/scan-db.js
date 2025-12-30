const { PrismaClient } = require('@prisma/client');
const prisma = new PrismaClient();

async function scan() {
    console.log("🔍 Scanning Database...\n");

    const users = await prisma.user.findMany({
        include: { routines: true }
    });

    console.log(`Found ${users.length} Users:`);
    users.forEach(u => {
        console.log(`- [${u.id}] ${u.email} (${u.firstName} ${u.lastName})`);
        if (u.routines.length > 0) {
            console.log(`   ✅ Routine Found: ${u.routines.length} routines`);
            u.routines.forEach(r => {
                console.log(`      - ID: ${r.id}, Name: ${r.name}, IsActive: ${r.isActive}`);
                console.log(`        Steps (Raw): ${r.steps}`);
            });
        } else {
            console.log(`   ❌ No Routines`);
        }
    });
}

scan()
    .catch(e => console.error(e))
    .finally(async () => await prisma.$disconnect());
