const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Mock data to simulate skinanalysisController output
const mockAnalysisResult = {
    skinScore: 85,
    skinType: 'Oily',
    detectedConditions: [
        { condition: 'Acne', confidence: 0.9, severity: 'Moderate' }
    ],
    scoreBreakdown: {
        hydration: 80,
        pigmentation: 90,
        texture: 70,
        clarity: 85
    },
    primaryCondition: {
        condition: 'Acne',
        confidence: 0.9,
        severity: 'Moderate'
    }
};

async function testPythonChain() {
    console.log("🚀 Testing Python Integration Chain...");

    // 1. Create Mock Output File (simulating step 1)
    const tempFile = path.join(os.tmpdir(), `test-analysis-${Date.now()}.json`);
    fs.writeFileSync(tempFile, JSON.stringify(mockAnalysisResult));
    console.log(`📄 Created mock analysis file: ${tempFile}`);

    // 2. Run Recommendation Controller
    const recScript = path.join(__dirname, '../src/controllers/recommendationController.py');
    const command = `python "${recScript}" --input "${tempFile}"`;

    console.log(`🤖 Running recommendation script: ${command}`);

    exec(command, (error, stdout, stderr) => {
        // Cleanup
        try { fs.unlinkSync(tempFile); } catch (e) { }

        if (error) {
            console.error("❌ Recommendation Script Failed:", error.message);
            if (stderr) console.error("Stderr:", stderr);
            return;
        }

        if (stderr) console.log("Stderr (Info):", stderr);

        // 3. Parse Output
        try {
            const startMarker = "--- JSON OUTPUT START ---";
            const endMarker = "--- JSON OUTPUT END ---";

            if (stdout.includes(startMarker) && stdout.includes(endMarker)) {
                const jsonStr = stdout.split(startMarker)[1].split(endMarker)[0].trim();
                const result = JSON.parse(jsonStr);

                console.log("\n✅ Recommendation Output Parsed Successfully!");
                if (result.recommendations && result.recommendations.products) {
                    console.log(`   Products Recommended: ${result.recommendations.products.length}`);
                    console.log("   Top Product:", result.recommendations.products[0].name);
                } else {
                    console.warn("⚠️ No recommendations found in output structure");
                    console.log(JSON.stringify(result, null, 2));
                }
            } else {
                console.error("❌ JSON Markers not found in output");
                console.log("Output:", stdout);
            }
        } catch (e) {
            console.error("❌ Failed to parse output:", e.message);
        }
    });
}

testPythonChain();
