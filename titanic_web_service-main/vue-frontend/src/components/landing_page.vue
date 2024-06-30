<template>
  <div class="mainWindow">
    <nav>
      <div class="nav-logo">
        <a href="#">
          <img src="../assets/ship.png">
        </a>
      </div>

      <ul class="nav-links">
        <li class="link"><a href="#team">About Group</a></li>
        <li class="link"><a href="#f-link">Links</a></li>
      </ul>
      <router-link to="/basics"><button class="btn">Learn Basics</button></router-link>
    </nav>

    <header class="container">
      <div class="content">
        <span class="blur"></span>
        <span class="blur"></span>
        <h4>Survive Better</h4>
        <H1>Get Survival<span> Predictions </span></H1>
        <p>
          This web application allows users to predict the survival chances of Titanic passengers based on various input
          parameters. Our goal is to showcase the application of machine learning models in a real-world scenario.

        </p>
        <router-link to="/predict"><button class="btn">Prediction Calculator</button></router-link>
      </div>
      <div class="image">
        <img src="../assets/header.png">
      </div>
    </header>

    <footer>
      <div id="team" class="footer-container">
        <h1>Meet Our Team</h1>
        <div class="team-grid">
          <div class="team-member">
            <img src="../assets/bhanu.png" alt="Bhanu Pratap Singh">
            <h4>Bhanu Pratap Singh</h4>
            <p>
            <ul>
              <li>Frontend Engineer</li>
              <li>UI/UX Designer</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/rohit.png" alt="Bhanu Pratap Singh">
            <h4>Rohit Galani</h4>
            <p>
            <ul>
              <li>Backend Engineer</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/nemish.png" alt="Bhanu Pratap Singh">
            <h4>Nemish Kyada</h4>
            <p>
            <ul>
              <li>Software Tester</li>
              <li>DevOps Engineer</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/aadi.png" alt="Bhanu Pratap Singh">
            <h4>Aadi Ahluwalia</h4>
            <p>
            <ul>
              <li>Deployment Specialist</li>
              <li>Version Control Manager</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/aryan.png" alt="Bhanu Pratap Singh">
            <h4>Aryan Jain</h4>
            <p>
            <ul>
              <li>Deployment Specialist</li>
              <li>Version Control Manager</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/avanish.png" alt="Bhanu Pratap Singh">
            <h4>Avanish Singh</h4>
            <p>
            <ul>
              <li>Software Tester</li>
              <li>DevOps Engineer</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/pp.webp" alt="Bhanu Pratap Singh">
            <h4>Harsh Gurawaliya</h4>
            <p>
            <ul>
              <li>Backend Engineer</li>
            </ul>
            </p>
          </div>
          <div class="team-member">
            <img src="../assets/pp.webp" alt="Bhanu Pratap Singh">
            <h4>Jijainth Dhinakaran</h4>
            <p>
            <ul>
              <li>Frontend Engineer</li>
              <li>UI/UX Designer</li>
            </ul>
            </p>
          </div>
        </div>
      </div>
      <div id="f-link" class="f-link">
        <h1>Relevant Links : </h1>
        <div class="f-links">
          <div class="gitlab">
            <h2>Gitlab Repository</h2>
            <img src="../assets/gitlab-logo.png" alt="">
          </div>
          <div class="dataset">
            <h2>Kaggle Dataset</h2>
            <img src="../assets/dataset.png" alt="">
          </div>
         
        </div>

      </div>
    </footer>

  </div>
</template>

<script>
export default {
  name: 'LandingPage'
}
</script>

<style scoped>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

* {
  padding: 0;
  margin: 0;
  outline: none;
  border: none;
  box-sizing: border-box;
}

.mainWindow {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  /* margin-top:-10vh; */
  font-family: 'Poppins', sans-serif;
  background-color: #18181B;
  /* padding-top: 2vh; */
}

img {
  width: 100%;
  display: flex;
}

a {
  text-decoration: none;
}

nav {
  max-width: 1200px;
  margin: auto;
  padding: 2rem 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
}

.nav-logo {
  max-width: 70px;
}

.nav-links {
  list-style: none;
  display: flex;
  align-items: center;
  gap: 3rem;
}

.link a {
  position: relative;
  padding-bottom: 0.75rem;
  color: #fff;
}

.link a::after {
  content: "";
  position: absolute;
  height: 2px;
  width: 0;
  bottom: 0;
  left: 0;
  background-color: #1d4ed8;
  transition: all 0.3s ease;
}

.link a:hover::after {
  width: 70%;
}

.btn {

  padding: 1rem 2rem;
  font-size: 1rem;
  color: #fff;
  background-color: #1d4ed8;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn:hover {
  background-color: #1e40af;
}

header .btn {
  margin-top: 2vh;
}

.container {
  max-width: 1200px;
  margin: auto;
  padding: 5rem 2rem;
}

header {
  position: relative;
  /* padding-top: 1rem; */
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  align-items: center;
  gap: 2rem;
}

header .content h4 {
  margin-bottom: 1rem;
  color: #ccc;
  font-size: 1rem;
  font-weight: 600;
}

header .content h1 {
  color: #fff;
  margin-bottom: 1rem;
  font-size: 3rem;
  font-weight: 700;
  line-height: 4rem;
}

header .content h1 span {
  -webkit-text-fill-color: transparent;
  -webkit-text-stroke: 1px #fff;
}

header .content p {
  margin-bottom: 2rem;
  color: #ccc;
}

header .image {
  position: relative;
}

header .image img {
  max-width: 600px;
  margin: auto;
}

footer {
  background-color: #1e1e21;
  color: #fff;
  padding: 2rem;
  width: 100%;
  text-align: center;
}

.footer-container {
  max-width: 1200px;
  margin: auto;
}

.footer-container h2 {
  font-size: 1.5rem;
  margin-bottom: 1.5rem;
}

.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 3rem;
}

.team-member {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 1rem;
}

.team-member img {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  object-fit: cover;
  margin-bottom: 0.5rem;
}

.team-member h4 {
  font-size: 1.2rem;
  margin-bottom: 0.5rem;
}

.team-member p {
  font-size: 0.7rem;
  color: #ccc;
}

.f-link {
  background-color: #18181B;
  margin: 4rem 0 0 0;
  padding: 2rem;
  /* border: 2px solid white; */
  border-radius: 50px;
}

.f-links {
  display: flex;
  flex-direction: row;
  justify-content: space-evenly;
  align-items: center;
  margin-top: 3rem;
}

.f-links img {
  width: 200px;
  height: 200px;
  border-radius: 10%;
  object-fit: contain;
  /* margin-bottom: 0.5rem; */
}

@media (width < 1000px) {
  .nav-links {
    display: flex;
    flex-direction: column;
  }

  header {
    grid-template-columns: repeat(1, 1fr);
  }

  header .image {
    grid-area: 1/1/2/2;

  }
}

@media (width < 600px) {
  header .image::before {
    display: none;
  }
}
</style>