import React, { useState } from 'react';
import './Table.css';  // Add custom styles here
import { Tooltip} from 'react-tooltip';


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

  const yearSet = new Set(data.map(item => item.year))
  return (
    <div className="table-container">
        <table className="table">
        <thead>
            <tr>
            <th data-tooltip-id="my-tooltip" data-tooltip-content="Index of the Record">Index</th>
            <th onClick={() => requestSort('name')} data-tooltip-id="my-tooltip" data-tooltip-content="Name of the Player">Name</th>
            <th onClick={() => requestSort('team')} data-tooltip-id="my-tooltip" data-tooltip-content="Team of the Player">Team</th>
            { yearSet.size > 1 && <th onClick={() => requestSort('year')} data-tooltip-id="my-tooltip" data-tooltip-content="Year of the record">Year</th>}
            <th onClick={() => requestSort('scoring')} data-tooltip-id="my-tooltip" data-tooltip-content="Scoring rating of the player, max is 1.0">Scoring</th>
            <th onClick={() => requestSort('playmaking')} data-tooltip-id="my-tooltip" data-tooltip-content="Playmaking rating of the player, max is 1.0">Playmaking</th>
            <th onClick={() => requestSort('rebounding')} data-tooltip-id="my-tooltip" data-tooltip-content="Rebounding rating of the player, max is 1.0">Rebounding</th>
            <th onClick={() => requestSort('defense')} data-tooltip-id="my-tooltip" data-tooltip-content="Defensive rating of the player, max is 1.0">Defense</th>
            <th onClick={() => requestSort('n_vorp')} data-tooltip-id="my-tooltip" data-tooltip-content="Value over replacement player, average player has a score of 0">Vorp</th>
            </tr>
        </thead>
        <tbody>
            {sortedData.map((row, index) => (
            <tr key={index}>
                <td>{index}</td>
                <td>{row.name}</td>
                <td>{row.team}</td>
                {yearSet.size > 1 && <td>{row.year}</td>}
                <td>{row.scoring}</td>
                <td>{row.playmaking}</td>
                <td>{row.rebounding}</td>
                <td>{row.defense}</td>
                <td>{row.n_vorp}</td>
            </tr>
            ))}
        </tbody>
        </table>
        <Tooltip id="my-tooltip" />

    </div>
  );
};

export default Table;
