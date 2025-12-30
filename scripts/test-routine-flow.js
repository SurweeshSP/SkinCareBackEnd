const BASE_URL = 'http://localhost:3000/api';

async function testRoutineFlow() {
    console.log("🚀 Starting Routine Flow Test...\n");

    // 1. Register/Login to get Token
    const email = `routine_tester_${Date.now()}@example.com`;
    console.log("1️⃣  Registering User...");
    const regRes = await fetch(`${BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email,
            password: "password123",
            firstName: "Routine",
            lastName: "Tester"
        })
    });
    const regData = await regRes.json();
    const token = regData.data?.token; // Adjust based on direct return or nested
    // Note: My authController correction returned nested data: { success: true, data: { token, user } }
    // But login returns flattened { token, user }. 
    // Wait, let's check exact response in regData.

    // Actually, I'll just use the token from register response.
    // AuthController.register returns: res.status(201).json({ success: true, data: { token, user } });

    const authToken = regData.data?.token;

    if (!authToken) {
        console.error("❌ Failed to get token:", regData);
        return;
    }
    console.log("✅ Token received!");

    // 2. Create/Update Routine
    console.log("\n2️⃣  Creating Routine...");
    const steps = [
        { id: '1', name: 'Cleanse', product: 'Soap', completed: false },
        { id: '2', name: 'Moisturize', product: 'Lotion', completed: true }
    ];

    const routineRes = await fetch(`${BASE_URL}/routines`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${authToken}`
        },
        body: JSON.stringify({
            name: "My Test Routine",
            frequency: "daily",
            steps: steps
        })
    });

    const routineData = await routineRes.json();
    console.log("   Status:", routineRes.status);
    console.log("   Response:", JSON.stringify(routineData, null, 2));

    if (routineRes.status === 201) {
        console.log("✅ Routine Created/Updated Successfully!");
    } else {
        console.error("❌ Routine Creation Failed!");
    }

    // 3. Fetch Routine
    console.log("\n3️⃣  Fetching Routine...");
    const getRes = await fetch(`${BASE_URL}/routines`, {
        headers: { 'Authorization': `Bearer ${authToken}` }
    });
    const getData = await getRes.json();
    console.log("   Status:", getRes.status);
    console.log("   Data:", JSON.stringify(getData, null, 2));
}

testRoutineFlow();
