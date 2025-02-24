const BACKEND_URL = "https://premai.onrender.com";  // ✅ Correct URL


function predictResidue() {
    const cShift = document.querySelector("#c_shift").value.trim();
    const caShift = document.querySelector("#ca_shift").value.trim();
    const cbShift = document.querySelector("#cb_shift").value.trim();

    if (!cShift || !caShift || !cbShift) {
        alert("⚠️ Please enter all chemical shifts before running analysis!");
        return;
    }

    const data = {
        C: parseFloat(cShift),
        CA: parseFloat(caShift),
        CB: parseFloat(cbShift),
    };

    fetch(`${BACKEND_URL}/predict`, { // Calls the Flask API
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        console.log(" API Response:", result);
        if (result.error) {
            document.querySelector("#output").innerHTML = ` Error: ${result.error}`;
        } else {
            let outputHTML = `<p> <b>Given Shifts:</b> C=${result.C}, CA=${result.CA}, CB=${result.CB}</p>`;
            outputHTML += "<p> <b>Predicted Residues:</b></p><ol>";
            result.Predictions.forEach(pred => {
                outputHTML += `<li> <b>${pred.Residue}</b> - ${pred.Probability * 100}%</li>`;
            });
            outputHTML += "</ol>";
            document.querySelector("#output").innerHTML = outputHTML;
        }
    })
    .catch(error => {
        document.querySelector("#output").innerHTML = " Failed to fetch prediction.";
        console.error("❌ Error:", error);
    });
}
