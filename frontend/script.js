document.getElementById("check-news").addEventListener("click", async function () {
    let newsText = document.getElementById("news-input").value;

    if (!newsText.trim()) {
        alert("Please enter a news article!");
        return;
    }

    console.log("Sending request to API...");

    try {
        let response = await fetch("http://127.0.0.1:8080/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: newsText })
        });

        let result = await response.json();
        console.log("Response received:", result);

        let resultBox = document.getElementById("result");
        resultBox.innerHTML = `<span>${result.prediction}</span>`;
        resultBox.style.color = result.prediction === "Real News" ? "green" : "red";

    } catch (error) {
        console.error("Error:", error);
        alert("Error connecting to API. Make sure the backend is running.");
    }
});