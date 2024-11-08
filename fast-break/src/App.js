import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import Table from './Table';
import './App.css';  // Add custom styles here
import TeamTable from './TeamTable';

const App = () => {
  const [data, setData] = useState([]);
  const [teams, setTeams] = useState([]);
  const [year, setYear] = useState("2024");
  const [searchQuery, setSearchQuery] = useState("");  // New state for search query

  useEffect(() => {
    // Fetch data from local file
    fetch('/players.json')
      .then(response => response.json())
      .then(data => setData(data))
      .catch(error => console.error('Error fetching data:', error));
    fetch('/teams.json')
      .then(response => response.json())
      .then(data => setTeams(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  let yearSet = new Set(data.map(item => item.year))
  yearSet.add('All Time (Worst Players)')
  yearSet.add('All Time (Best Players)')
  const years = [...yearSet]
  const yearOptions = years.map(year => ({ value: year, label: year }));

  const handleYearChange = (selectedOption) => {
    setYear(selectedOption.value);
  };

  let prefilteredData = data.filter(item => item.year === year || year === 'All Time (Worst Players)' || year === 'All Time (Best Players)');
  if (year === 'All Time (Worst Players)'){
    prefilteredData.sort((a, b) => a.n_vorp - b.n_vorp)
  } else {
    prefilteredData.sort((a, b) => b.n_vorp - a.n_vorp)
  }
  if (prefilteredData.length > 1000){
    prefilteredData = prefilteredData.slice(0, 1000)
  }
  const filteredData = prefilteredData.filter(item => item.name.toLowerCase().includes(searchQuery.toLowerCase()))
  const filteredTeams = teams.filter(item => item.year === year);

  return (
    <div className="App">
      <h1>Fast Break</h1>
      <Select
        options={yearOptions}
        onChange={handleYearChange}
        defaultValue={yearOptions.find(option => option.value === year)}
      />
      <input
        type="text"
        placeholder="Search by name"
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
        className="search-bar"  // Optional: Add a CSS class for styling
      />
      <Table data={filteredData} />
      <TeamTable data={filteredTeams} />
      <p className="Tex">FastBreak is a statistical analysis tool that ranks players and teams based on their attributes.
      There are four major attributes to a player's prowess: playmaking, rebounding, scoring, and defense.</p>
      <p className="Tex">
      The average in any of these statistics is 0.0, meaning an average player has a rating of 0, the score is a representation of how many standard deviations away form the average a player is.
      These statistics, excluding defense, follow a simple formula: efficiency + (1.5 * volume). 
      We find the efficiency and volume for each player in each of those stats, then find the z-score of those statistics and plug it into the aforementioned 
      formula to obtain their rating.</p>
      <p className="Tex">
      Vorp, or value over replacement player, represents how many standard deviations of value a player is from the average. This is calculated by combinging the other four statistics
      Therefore, a vorp of 0 is an average player, while a score of 4 means you are four standard deviations from the average player.
      </p>
      <p className="Tex">
      We plug the playmaking, scoring, defense, and rebounding statistics into neural network in order to predict team rankings. The neural network operates within a 10% accuracy range
      </p>
      <div>
        <a href="https://github.com/Pixelgost/FastBreak" target="_blank" rel="noopener noreferrer" className="github-button">
           Link to Github Repository
        </a>
       </div>
        <a href="https://www.basketball-reference.com/" target="_blank" rel="noopener noreferrer" className='italic'>
           All statistics are pulled from basketball reference
        </a>
    </div>
    
  );
};

export default App;
