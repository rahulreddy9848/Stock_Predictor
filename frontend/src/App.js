import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);

function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [daysToPredict, setDaysToPredict] = useState(5);
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);

  useEffect(() => {
    if (predictions && chartRef.current) {
      renderChart();
    }
  }, [predictions]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setPredictions(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ ticker, days_to_predict: daysToPredict }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Something went wrong with the prediction.');
      }

      const data = await response.json();
      setPredictions(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderChart = () => {
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const ctx = chartRef.current.getContext('2d');

    const historicalDates = predictions.historical_data.map(item => item.Date);
    const historicalPrices = predictions.historical_data.map(item => item.Close);

    const predictionDates = predictions.predictions.map(item => item.Date);
    const predictionPrices = predictions.predictions.map(item => item['Predicted Close']);

    const allDates = [...historicalDates, ...predictionDates];
    const allPrices = [...historicalPrices, ...predictionPrices];

    chartInstance.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allDates,
        datasets: [
          {
            label: 'Historical Prices',
            data: historicalPrices,
            borderColor: 'rgb(75, 192, 192)',
            tension: 0.1,
            fill: false,
          },
          {
            label: 'Predicted Prices',
            data: Array(historicalDates.length).fill(null).concat(predictionPrices),
            borderColor: 'rgb(255, 99, 132)',
            borderDash: [5, 5],
            tension: 0.1,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          title: {
            display: true,
            text: `Stock Price Prediction for ${predictions.ticker}`,
          },
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Date',
            },
          },
          y: {
            title: {
              display: true,
              text: 'Price (USD)',
            },
          },
        },
      },
    });
  };

  return (
    <div className="App">
      <header className="App-header bg-dark text-white py-4">
        <h1 className="display-4">Stock Price Predictor</h1>
      </header>
      <main className="container mt-5">
        <div className="row justify-content-center">
          <div className="col-md-8">
            <div className="card shadow-lg p-4">
              <h2 className="card-title text-center mb-4">Predict Stock Prices</h2>
              <form onSubmit={handleSubmit}>
                <div className="mb-3">
                  <label htmlFor="tickerInput" className="form-label">Stock Ticker:</label>
                  <input
                    type="text"
                    className="form-control"
                    id="tickerInput"
                    value={ticker}
                    onChange={(e) => setTicker(e.target.value)}
                    required
                  />
                </div>
                <div className="mb-3">
                  <label htmlFor="daysInput" className="form-label">Days to Predict:</label>
                  <input
                    type="number"
                    className="form-control"
                    id="daysInput"
                    value={daysToPredict}
                    onChange={(e) => setDaysToPredict(e.target.value)}
                    min="1"
                    required
                  />
                </div>
                <button type="submit" className="btn btn-primary w-100" disabled={loading}>
                  {loading ? 'Predicting...' : 'Get Predictions'}
                </button>
              </form>

              {error && (
                <div className="alert alert-danger mt-4" role="alert">
                  Prediction Error: {error}
                </div>
              )}

              {predictions && (
                <div className="mt-4">
                  <h3 className="text-center mb-3">Stock Price Chart</h3>
                  <canvas ref={chartRef} id="stockChart"></canvas>
                </div>
              )}

              {predictions && predictions.predictions.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-center mb-3">Predicted Prices</h3>
                  <table className="table table-striped table-bordered">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Predicted Close Price</th>
                      </tr>
                    </thead>
                    <tbody>
                      {predictions.predictions.map((prediction, index) => (
                        <tr key={index}>
                          <td>{prediction.Date}</td>
                          <td>{prediction['Predicted Close'].toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
      <footer className="App-footer bg-dark text-white py-3 mt-5">
        <p>&copy; 2024 Stock Price Predictor</p>
      </footer>
    </div>
  );
}

export default App;
