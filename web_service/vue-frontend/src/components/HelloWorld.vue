<template>
  <div id="app">
    <header>
        <h1>Titanic Survivor Prediction App</h1>
        <nav>
          <router-link to="/"><button class="btn">Home</button></router-link>
          
            <ul>
                <li class="link"><a href="#introduction">Introduction</a></li>
                <li class="link"><a href="#technical-explanation">Technical Explanation</a></li>
                <li class="link"><a href="#user-guide">User Guide</a></li>
                <li class="link"><a href="#team">Team</a></li>
            </ul>   
            <router-link to="/predict"><button class="btn">Prediction Calculator</button></router-link>
        </nav>
    </header>

    <main>
        <section id="introduction">
            <h2>Project Overview</h2>
            <p>This project aims to develop a web application to predict the survival chances of passengers in the Titanic disaster. The app uses multiple machine learning models to provide accurate predictions based on user input.</p>
        </section>

        <section id="technical-explanation">
            <h2>How the Code Works</h2>
            <div class="technical-details">
                <h3>Backend:</h3>
                <p>Developed using FastAPI for the web framework. Handles user inputs and communicates with the prediction model service.</p>
                <h3>Frontend:</h3>
                <p>Built using VueJS, JavaScript, HTML, and CSS. Provides an interactive and responsive user interface for entering passenger details and viewing predictions.</p>
                <h3>Prediction Models:</h3>
                <p>Models include Random Forest, Decision Tree, KNN, Support Vector Machines, and Logistic Regression. Models are trained and stored as Pickle files, accessed via a RESTful API.</p>
                <h3>Integration:</h3>
                <p>Dockerized services ensure seamless integration and deployment. Docker Compose is used to manage multi-container applications. Continuous integration and deployment are handled through GitLab CI.</p>
            </div>
        </section>

        <section id="user-guide">
            <h2>User Guide</h2>
            <div class="guide-details">
                <h3>Building and Running the App:</h3>
                <p>Instructions on setting up the development environment.</p>
                <p>Steps to build and run the application using Docker Compose.</p>
                <h3>Accessing the Web Application:</h3>
                <p>How to access the application at <a href="https://localhost:8080">https://localhost:8080</a>.</p>
            </div>
        </section>

        <section id="team">
            <h2>The Team Behind the Project</h2>
            <div class="team-member">
                <h3>Bhanu Pratap Singh</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Rohit Galani</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Nemish Kyada</h3>
                <p><strong>Matriculation Number:</strong> 22212034</p>
                <p><strong>Role:</strong> Software Tester</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Aadi Ahluwalia</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Aryan Jain</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Avanish Singh</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Harhs Gurwaliya</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
            <div class="team-member">
                <h3>Jijainth Dhinakaran</h3>
                <p><strong>Matriculation Number:</strong> 123456</p>
                <p><strong>Role:</strong> Frontend Developer</p>
                <p><strong>Contribution:</strong> Designed the UI/UX of the web application and implemented the frontend using VueJS.</p>
            </div>
        </section>
    </main>

    <footer>
       
    </footer>
  </div>
</template>

<script>

export default {
  name: 'App'
}
</script>

<style scoped>

*{
    padding: 0;
    margin: 0;
    outline: none;
    border: none;
    box-sizing: border-box;
    color: #fff;
}

#app{
  font-family: 'Poppins', sans-serif;
  background-color: #18181B;
  padding-top: 2vh;
  color: #fff;

}

header {
  background-color: #1f2937;
  padding: 2rem;
  display: flex;
  flex-direction: column;
}

h1 {
  text-align: center;
  margin-bottom: 1.5rem;
}

nav{
  display: flex;
  flex-direction: row;
  justify-content:space-between ;
}

nav ul {
  height: 3.5rem;
  width: 50%;
  border-radius: 10rem;
  background: rgba(0, 0, 255, 0.128);
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 1rem;
}

.link a{
    position: relative;
    padding-bottom: 0.75rem;
    color: #fff;
    font-size: 1rem;
}

.link a::after{
    content: "";
    position: absolute;
    height: 2px;
    width: 0;
    bottom: 0;
    left: 0;
    background-color: #1d4ed8;
    transition: all 0.3s ease;
}

.link a:hover::after{
    width: 70%;
}

nav ul li {
  list-style: none;
}

a{
    text-decoration: none;
}

.btn{
  width: 15rem;
  padding: 1rem 2rem;
  font-size: 1rem;
  color: #fff;
  background-color: #1d4ed8;
  border-radius: 50px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn:hover{
    background-color: #1e40af;
}

main {
  display: flex;
  flex-direction: column;
  justify-items: center;
  align-items: center;
  padding: 2rem;
  margin: 1rem;
}

section {
  margin-bottom: 2rem;
  width: 80%;
}

h2 {
  background-color: rgba(0, 0, 255, 0.128);
  height: 3.5rem;
  width:auto;
  font-size: 1.5rem;
  color: #ffffff;
  margin-bottom: 2rem;
  padding-top: 0.6rem;
  border-radius: 10rem;
}

p {
  line-height: 1.7;
}

.technical-details, .guide-details {
  background-color: #18181B;
  padding: 1rem;
  border-radius: 8px;
}

.team-member {
  background-color: #18181B;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

footer {
  background-color: #18181B;
  text-align: center;
  padding: 1rem;
  margin-top: 2rem;
}

</style>
