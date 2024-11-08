import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const ScoreChart = ({ players }) => {
  // Prepare data for the chart with each metric as a separate dataset
  const data = {
    labels: players.map(player => `${parseInt(player.year)-1}-${player.year.slice(-2)}`), // years on x-axis
    datasets: [
      {
        label: 'VORP',
        data: players.map(player => player.n_vorp),
        borderColor: 'rgba(173, 216, 230, 1)', // light blue
        backgroundColor: 'rgba(173, 216, 230, 0.2)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Scoring',
        data: players.map(player => player.scoring),
        borderColor: 'rgba(0, 128, 0, 1)', // green
        backgroundColor: 'rgba(0, 128, 0, 0.2)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Playmaking',
        data: players.map(player => player.playmaking),
        borderColor: 'rgba(255, 255, 0, 1)', // yellow
        backgroundColor: 'rgba(255, 255, 0, 0.2)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Rebounding',
        data: players.map(player => player.rebounding),
        borderColor: 'rgba(139, 69, 19, 1)', // brown
        backgroundColor: 'rgba(139, 69, 19, 0.2)',
        fill: false,
        tension: 0.4,
      },
      {
        label: 'Defense',
        data: players.map(player => player.defense),
        borderColor: 'rgba(0, 0, 0, 1)', // black
        backgroundColor: 'rgba(0, 0, 0, 0.2)',
        fill: false,
        tension: 0.4,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: `${players[0].name} Performance Over Time` },
    },
  };

  return <Line data={data} options={options} />;
};

export default ScoreChart;
