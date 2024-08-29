import React, { useState } from 'react';
import './Table.css';  // Add custom styles here

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
            <th onClick={() => requestSort('team')}>Team Name</th>
            <th onClick={() => requestSort('predicted_win_rate')}>Predicted Win Rate</th>
            <th onClick={() => requestSort('actual_win_rate')}>Actual Win Rate</th>
            <th onClick={() => requestSort('predicted_wins')}>Predicted Wins</th>
            <th onClick={() => requestSort('actual_wins')}>Actual Wins</th>
            </tr>
        </thead>
        <tbody>
            {sortedData.map((row, index) => (
            <tr key={index}>
                <td>{row.team}</td>
                <td>{row.predicted_win_rate}</td>
                <td>{row.actual_win_rate}</td>
                <td>{row.predicted_wins}</td>
                <td>{row.actual_wins}</td>
            </tr>
            ))}
        </tbody>
        </table>
      </div>
    }
  </div>
  );
};

export default TeamTable;