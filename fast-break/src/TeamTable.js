import React, { useState, useEffect } from 'react';
import './Table.css';  // Add custom styles here
import { Tooltip } from 'react-tooltip';
import Modal from 'react-modal';
import * as ort from 'onnxruntime-web';

Modal.setAppElement('#root');


const TeamTable = ({ data, playerList }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'team', direction: 'ascending' });
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [playerStats, setPlayerStats] = useState([])
  const [model, setModel] = useState(null);
  
  useEffect(() => {
    const loadModel = async () => {
      try {
        const session = await ort.InferenceSession.create('/model.onnx'); // Path to your model in the public folder
        setModel(session);
      } catch (error) {
        console.error('Failed to load model:', error);
      }
    };
    loadModel();
  }, []);

  const openModal = (team) => {
    setSelectedTeam(team);
    setPlayerStats(playerList.filter((player) => player.team === team.team).sort((a, b) => b.n_vorp - a.n_vorp).slice(0, 8))
    console.log(team)
    console.log(playerStats)
    setIsOpen(true);
  };

  const closeModal = () => {
    setIsOpen(false);
    setPlayerStats([])
    setSelectedTeam(null);
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
                <td>
                <button className="player-button" onClick={() => openModal(row)}>{row.team}</button>
                </td>
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
              width: '80%',
              height: '80%',
              marginRight: '-50%',
              transform: 'translate(-50%, -50%)',
            },
          }}
        >
          <h2>{selectedTeam?.team}'s Predictor</h2>
          <button onClick={closeModal}>Close</button>
          {playerStats && playerStats.length > 0 && playerStats.map((player, index) => {
              return(
              <div key={index}>
                <p>{player.name}</p>
                <label>Playmaking: {playerStats[index].playmaking}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.2}
                  value={playerStats[index].playmaking}
                  onChange={(e) => { 
                    const arr = [...playerStats]
                    arr[index].playmaking = parseFloat(e.target.value);
                    setPlayerStats(arr)
                  }}
                />
                <label>Scoring: {playerStats[index].scoring}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].scoring}
                  onChange={(e) => { 
                    const arr = [...playerStats]
                    arr[index].scoring = parseFloat(e.target.value);
                    setPlayerStats(arr)
                  }}
                />
                <label>Rebounding: {playerStats[index].rebounding}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].rebounding}
                  onChange={(e) => { 
                    const arr = [...playerStats]
                    arr[index].rebounding = parseFloat(e.target.value);
                    setPlayerStats(arr)
                  }}
                />
                <label>Defense: {playerStats[index].defense}</label>
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].defense}
                  onChange={(e) => { 
                    const arr = [...playerStats]
                    arr[index].defense = parseFloat(e.target.value);
                    setPlayerStats(arr)
                  }}
                />
                <label>Games Played: {playerStats[index].games_played}</label>
                <input
                  type="range"
                  min={0}
                  max={82}
                  value={playerStats[index].games_played}
                  onChange={(e) => { 
                    const arr = [...playerStats]
                    arr[index].games_played = e.target.value;
                    setPlayerStats(arr)
                  }}
                />
              </div>
              )
          })}
          <button className='player-button' onClick={async (e) => {
            const data = [];
            playerStats.forEach((item) => {
              data.push(item.scoring, item.playmaking, item.rebounding, item.defense, (parseInt(item.games_played) / 82.0))
            })
            console.log(data)
            if (data.length === 40) {
              const inputTensor = new ort.Tensor('float32', data, [1, data.length]); // Adjust dimensions as needed
              const feeds = { input: inputTensor }; // Replace input_name with your model's input name
              const output = await model.run(feeds);
              if (output.output.cpuData.length > 0) {
                const rate = output.output.cpuData[0]
                const wins = Math.round(82 * rate)
                const losses = 82 - wins
                alert(`Predicted Record: ${wins} - ${losses}`)
              } else {
                alert("Internal Error!")
              }
            } else {
              alert("Not Enough Data!")
            }
          }}>
            Submit
          </button>
      </Modal>
  </div>
  );
};

export default TeamTable;
