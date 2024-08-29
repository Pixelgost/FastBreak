import React, { useState } from 'react';
import './Table.css';  // Add custom styles here

const Table = ({ data }) => {
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

  return (
    <div className="table-container">
        <table className="table">
        <thead>
            <tr>
            <th onClick={() => requestSort('name')}>Name</th>
            <th onClick={() => requestSort('scoring')}>Scoring</th>
            <th onClick={() => requestSort('playmaking')}>Playmaking</th>
            <th onClick={() => requestSort('rebounding')}>Rebounding</th>
            <th onClick={() => requestSort('defense')}>Defense</th>
            <th onClick={() => requestSort('vorp')}>Vorp</th>
            <th onClick={() => requestSort('n_vorp')}>Standardized Vorp</th>
            </tr>
        </thead>
        <tbody>
            {sortedData.map((row, index) => (
            <tr key={index}>
                <td>{row.name}</td>
                <td>{row.scoring}</td>
                <td>{row.playmaking}</td>
                <td>{row.rebounding}</td>
                <td>{row.defense}</td>
                <td>{row.vorp}</td>
                <td>{row.n_vorp}</td>
            </tr>
            ))}
        </tbody>
        </table>
    </div>
  );
};

export default Table;
