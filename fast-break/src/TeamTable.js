import React, { useState, useEffect } from 'react';
import './Table.css';  // Add custom styles here
import { Tooltip } from 'react-tooltip';
import Modal from 'react-modal';
import './styles.css';
import * as ort from 'onnxruntime-web';

Modal.setAppElement('#root');

const TeamTable = ({ data, playerList }) => {
  const [sortConfig, setSortConfig] = useState({ key: 'team', direction: 'ascending' });
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTeam, setSelectedTeam] = useState(null);
  const [playerStats, setPlayerStats] = useState([]);
  const [originalPlayerStats, setOriginalPlayerStats] = useState([])
  const [model, setModel] = useState(null);
  const [hideSliders, setHideSliders] = useState(true); // State for checkbox
  const scalingEffect = 1.5
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
    setOriginalPlayerStats(JSON.parse(JSON.stringify(playerList.filter((player) => player.team === team.team).sort((a, b) => b.n_vorp - a.n_vorp).slice(0, 8))))
    setIsOpen(true);
  };

  const closeModal = () => {
    setIsOpen(false);
    setPlayerStats([]);
    setOriginalPlayerStats([]);
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

  const shouldShowTable = data.length > 0;

  const handleGamesPlayedChange = (e, index) => {
    const newGamesPlayed = e.target.value;
    // Update other stats proportionally to the scaling factor (60% of the change)
    const updatedStats = [...playerStats];
    const scalingFactor = (newGamesPlayed / originalPlayerStats[index].games_played); // Calculate how much Games Played has changed
    updatedStats[index].games_played = newGamesPlayed;
    setPlayerStats(updatedStats);
    if (hideSliders) {
      updatedStats[index].scoring = originalPlayerStats[index].scoring * (scalingFactor * scalingEffect);
      updatedStats[index].playmaking = originalPlayerStats[index].playmaking * (scalingFactor * scalingEffect);
      updatedStats[index].defense = originalPlayerStats[index].defense * (scalingFactor * scalingEffect);
      updatedStats[index].rebounding = originalPlayerStats[index].rebounding * (scalingFactor * scalingEffect);
      updatedStats[index].games_played = newGamesPlayed;
    }
    
    setPlayerStats(updatedStats);
    
  };

  return (
    <div>
      {shouldShowTable &&
        <div className="table-container">4
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
                <th onClick={() => requestSort('actual_win_rate')} data-tooltip-id="my-tooltip" data-tooltip-content="Actual Record of the Teams">Actual Record</th>
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
            width: '60%',
            height: '80%',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
          },
          overlay: {
            backgroundColor: 'rgba(0, 0, 0, 0.5)', // Semi-transparent overlay
          },
        }}
      >
        <button onClick={closeModal}>
          &times;
        </button>
        <h2 className="modal-header">{selectedTeam?.team}'s Predictor</h2>

        {/* Checkbox to toggle sliders */}
        <div>
          <label>
            <input 
              type="checkbox" 
              checked={hideSliders}
              onChange={() => setHideSliders(!hideSliders)} 
            />
            Auto-Scale With Games Played
          </label>
        </div>

        {playerStats && playerStats.length > 0 && playerStats.map((player, index) => (
          <div key={index} className="player-stats">
            <h3>{player.name}</h3>
            <div>
              <label className="range-label">Playmaking: {playerStats[index].playmaking}</label>
              {!hideSliders && (
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.2}
                  value={playerStats[index].playmaking}
                  onChange={(e) => {
                    const arr = [...playerStats];
                    arr[index].playmaking = parseFloat(e.target.value);
                    setPlayerStats(arr);
                  }}
                  className="range-input"
                />
              )}
            </div>
            <div>
              <label className="range-label">Scoring: {playerStats[index].scoring}</label>
              {!hideSliders && (
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].scoring}
                  onChange={(e) => {
                    const arr = [...playerStats];
                    arr[index].scoring = parseFloat(e.target.value);
                    setPlayerStats(arr);
                  }}
                  className="range-input"
                />
              )}
            </div>
            <div>
              <label className="range-label">Rebounding: {playerStats[index].rebounding}</label>
              {!hideSliders && (
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].rebounding}
                  onChange={(e) => {
                    const arr = [...playerStats];
                    arr[index].rebounding = parseFloat(e.target.value);
                    setPlayerStats(arr);
                  }}
                  className="range-input"
                />
              )}
            </div>
            <div>
              <label className="range-label">Defense: {playerStats[index].defense}</label>
              {!hideSliders && (
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={playerStats[index].defense}
                  onChange={(e) => {
                    const arr = [...playerStats];
                    arr[index].defense = parseFloat(e.target.value);
                    setPlayerStats(arr);
                  }}
                  className="range-input"
                />
              )}
            </div>
            <div>
              <label className="range-label">Games Played: {playerStats[index].games_played}</label>
              <input
                type="range"
                min={1}
                max={82}
                value={playerStats[index].games_played}
                onChange={(e) => handleGamesPlayedChange(e, index)}
                className="range-input"
              />
            </div>
          </div>
        ))}

        <button
          className="button-submit"
          onClick={async (e) => {
            const data = [];
            playerStats.forEach((item) => {
              data.push(item.scoring, item.playmaking, item.rebounding, item.defense, (parseInt(item.games_played) / 82.0));
            });
            console.log(data);
            if (data.length === 40) {
              const inputTensor = new ort.Tensor('float32', data, [1, data.length]);
              const feeds = { input: inputTensor };
              const output = await model.run(feeds);
              if (output.output.cpuData.length > 0) {
                const rate = output.output.cpuData[0];
                const wins = Math.max(Math.min(Math.round(82 * rate), 82), 0);
                const losses = 82 - wins;
                alert(`Predicted Record: ${wins} - ${losses}`);
              } else {
                alert("Internal Error!");
              }
            } else {
              alert("Not Enough Data!");
            }
          }}
        >
          Submit
        </button>
      </Modal>
    </div>
  );
};

export default TeamTable;
