fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text: userReview }),
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    // 結果をUIに表示する
});
