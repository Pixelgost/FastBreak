import React, { useState } from 'react';
import './Table.css';  // Add custom styles here
import { Tooltip} from 'react-tooltip';
import Modal from 'react-modal';
import ScoreChart from './PlayerChart';

Modal.setAppElement('#root');

const Table = ({ data, playerList }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'team', direction: 'ascending' });
  const [isOpen, setIsOpen] = useState(false);
  const [selectedPlayer, setSelectedPlayer] = useState(null);


  const openModal = (player) => {
    setSelectedPlayer(player);
    setIsOpen(true);
  };

  const closeModal = () => {
    setIsOpen(false);
    setSelectedPlayer(null);
  };
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
            <th onClick={() => requestSort('scoring')} data-tooltip-id="my-tooltip" data-tooltip-content="Scoring rating of the player, average player has a score of 0">Scoring</th>
            <th onClick={() => requestSort('playmaking')} data-tooltip-id="my-tooltip" data-tooltip-content="Playmaking rating of the player, average player has a score of 0">Playmaking</th>
            <th onClick={() => requestSort('rebounding')} data-tooltip-id="my-tooltip" data-tooltip-content="Rebounding rating of the player, average player has a score of 0">Rebounding</th>
            <th onClick={() => requestSort('defense')} data-tooltip-id="my-tooltip" data-tooltip-content="Defensive rating of the player, average player has a score of 0">Defense</th>
            <th onClick={() => requestSort('n_vorp')} data-tooltip-id="my-tooltip" data-tooltip-content="Value over replacement player, average player has a score of 0">Vorp</th>
            </tr>
        </thead>
        <tbody>
            {sortedData.map((row, index) => (
            <tr key={index}>
                <td>{index + 1}</td>
                <td>
                  <button className="player-button" onClick={() => openModal(row)}>{row.name}</button>
                </td>
                <td>{row.team}</td>
                {yearSet.size > 1 && <td>{row.year}</td>}
                <td>{row.scoring}</td>
                <td>{row.playmaking > -1000 ? row.playmaking : 'NaN'}</td>
                <td>{row.rebounding}</td>
                <td>{row.defense}</td>
                <td>{row.n_vorp > -1000 ? row.n_vorp : 'NaN'}</td>
            </tr>
            ))}
        </tbody>
        </table>
        <Tooltip id="my-tooltip" />
        <Modal
          isOpen={isOpen}
          onRequestClose={closeModal}
          contentLabel="Player Chart"
          style={{
            content: {
              top: '50%',
              left: '50%',
              right: 'auto',
              bottom: 'auto',
              width: '50%',
              height: '50%',
              marginRight: '-50%',
              transform: 'translate(-50%, -50%)',
            },
          }}
        >
          <button onClick={closeModal}>&times;</button>
          <h2>{selectedPlayer?.name}'s Performance</h2>
          {selectedPlayer && (
            <ScoreChart className="player-button" players={playerList.filter(p => p.name === selectedPlayer.name)} />
          )}
      </Modal>
    </div>
  );
};

export default Table;
