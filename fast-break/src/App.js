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
      <Table data={filteredData} playerList={data} />
      <TeamTable data={filteredTeams} playerList={filteredData} />
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

      <p className="Tex">
      You can view a players's improvements over time by clicking on their name, you can also deselect stats you do not want to see.
      Keep in mind these are in terms of standard deviations so they are comparable over the years
      </p>
      <p className="Tex">
      You can try and predict a team's record by clicking on their name and adjusting their player's stats. 
      Keep in mind that games played is out of 82 so if we are only halfway through the season, adjust the games played to be all the way through.
      </p>
      <p className='Tex'>
      Also since volume affects a player's score for all stats, if you want to increase a player's games played, use the auto-scale feature to also adjust the stats to be a good estimate.
      </p>
      <p className='Tex'>
      If a player has the same stats as another player with less games played, the player with less games played will be counted as better. This is to say, if you don't scale stats with the games played the record will be worse as games played increases
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
