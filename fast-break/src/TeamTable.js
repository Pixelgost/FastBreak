import React, { useState } from 'react';
import './Table.css';  // Add custom styles here
import { Tooltip } from 'react-tooltip';
const TeamTable = ({ data }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'team', direction: 'ascending' });

  const sortedData = [...data].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? -1 : 1;
    }
    if (a[sortConfig.key] > b[sortConfig.key]) {
      return sortConfig.direction === 'ascending' ? 1 : -1;
    }
    return 0;
  });

  const requestSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };
  const shouldShowTable = data.length > 0
  return (
    <div>
      {shouldShowTable &&
      <div className="table-container">
      
        <table className="table">
        <thead>
            <tr>
            <th data-tooltip-id="my-tooltip" data-tooltip-content="Index of the Record">Index</th>
            <th onClick={() => requestSort('team')} data-tooltip-id="my-tooltip" data-tooltip-content="Name of the Team">Team Name</th>
            <th onClick={() => requestSort('predicted_win_rate')} data-tooltip-id="my-tooltip" data-tooltip-content="Neural network predicted win rate">Predicted Win Rate</th>
            <th onClick={() => requestSort('predicted_loss_rate')} data-tooltip-id="my-tooltip" data-tooltip-content="Neural network predicted loss rate">Predicted Loss Rate</th>
            <th onClick={() => requestSort('actual_win_rate')} data-tooltip-id="my-tooltip" data-tooltip-content="Actual win rate">Actual Win Rate</th>
            <th onClick={() => requestSort('actual_loss_rate')} data-tooltip-id="my-tooltip" data-tooltip-content="Actual loss rate">Actual Loss Rate</th>
            <th onClick={() => requestSort('predicted_wins')} data-tooltip-id="my-tooltip" data-tooltip-content=" Neural Network predicted record">Predicted Record</th>
            <th onClick={() => requestSort('actual_wins')} data-tooltip-id="my-tooltip" data-tooltip-content="Actual Record of the Teams">Actual Record</th>
            </tr>
        </thead>
        <tbody>
            {sortedData.map((row, index) => (
            <tr key={index}>
                <td>{index + 1}</td>
                <td>{row.team}</td>
                <td>{row.predicted_win_rate}</td>
                <td>{row.predicted_loss_rate}</td>
                <td>{row.actual_win_rate}</td>
                <td>{row.actual_loss_rate}</td>
                <td>{`${row.predicted_wins} - ${row.predicted_losses}`}</td>
                <td>{`${row.actual_wins} - ${row.actual_losses}`}</td>

            </tr>
            ))}
        </tbody>
        </table>
        <Tooltip id="my-tooltip" />
      </div>
    }
  </div>
  );
};

export default TeamTable;
