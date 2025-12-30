// Native fetch is verified available in Node v22+

const BASE_URL = 'http://localhost:3000/api';

async function testFlow() {
    console.log("🚀 Starting API Flow Test...\n");

    // 1. Register
    console.log("1️⃣  Registering User...");
    const regRes = await fetch(`${BASE_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email: `test_${Date.now()}@example.com`,
            password: "password123",
            firstName: "API",
            lastName: "Tester"
        })
    });
    const regData = await regRes.json();
    console.log("   Status:", regRes.status);
    console.log("   Response:", JSON.stringify(regData, null, 2));

    if (!regRes.ok) return;

    // 2. Login
    console.log("\n2️⃣  Logging In...");
    const loginRes = await fetch(`${BASE_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            email: regData.data.user.email,
            password: "password123"
        })
    });
    const loginData = await loginRes.json();
    console.log("   Status:", loginRes.status);

    const token = loginData.data?.token;
    if (!token) {
        console.error("❌ No token received!");
        return;
    }
    console.log("✅ Token received!");

    // 3. Create Analysis (Protected Route)
    console.log("\n3️⃣  Creating Analysis (With Token)...");
    const analysisRes = await fetch(`${BASE_URL}/analyses/create`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`  // <--- KEY PART
        },
        body: JSON.stringify({
            imageUrl: "test_image_key_from_api_test"
        })
    });
    const analysisData = await analysisRes.json();
    console.log("   Status:", analysisRes.status);
    console.log("   Response:", JSON.stringify(analysisData, null, 2));
}

testFlow();
