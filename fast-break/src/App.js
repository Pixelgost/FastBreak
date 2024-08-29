import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import Table from './Table';
import './App.css';  // Add custom styles here
import TeamTable from './TeamTable';

const App = () => {
  const [data, setData] = useState([]);
  const [teams, setTeams] = useState([]);
  const [year, setYear] = useState("2024");
  
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

  const years = [...new Set(data.map(item => item.year))]
  const yearOptions = years.map(year => ({ value: year, label: year }));

  const handleYearChange = (selectedOption) => {
    setYear(selectedOption.value);
  };

  const filteredData = data.filter(item => item.year === year);
  const filteredTeams = teams.filter(item => item.year === year);

  return (
    <div className="App">
      <h1>FastBreak</h1>
      <Select
        options={yearOptions}
        onChange={handleYearChange}
        defaultValue={yearOptions.find(option => option.value === year)}
      />
      <Table data={filteredData} />
      <TeamTable data={filteredTeams} />
      <p className="Tex">FastBreak is a statistical analysis tool that ranks players and teams based on their attributes.
      There are four major attributes to a player's prowess: playmaking, rebounding, scoring, and defense.</p>
      <p className="Tex">
      The maximum in any of these statistics is 1.0, meaning the best player has a rating of 1, while everyone else has a fraction of the best player.
      These statistics, excluding defense, follow a simple formula: efficiency + (1.5 * volume). 
      We find the efficiency and volume for each player in each of those stats, then find the z-score of those statistics and plug it into the aforementioned 
      formula to obtain their rating. Standardizing these statistics helps us compare these players to an 'average player'.</p>
      <p className="Tex">
      Vorp is calculated using a formula of those statistics. Standardized vorp represents how many standard deviations in vorp a player is from the average.
      Therefore, a standardized vorp of 0 is an average player, while a score of 4 means you are four standard deviations from the average player.
      We plug these calculated values into a neural network and predict team rankings.</p>
        <a href="https://github.com/Pixelgost/FastBreak" target="_blank" rel="noopener noreferrer" className="github-button">
           Link to Github Repository
        </a>
    </div>
    
  );
};

export default App;
