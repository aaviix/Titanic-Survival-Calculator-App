<template>
  <div class="main">
    <header>
      <nav>
        <router-link to="/"><button class="btn">Home</button></router-link>
        <h1 class="text">Titanic Survivor Prediction App</h1>
        <router-link to="/basics"><button class="btn">Learn Basics</button></router-link>
      </nav>
    </header>

    <div class="explain">
      Lorem ipsum, dolor sit amet consectetur adipisicing elit. Assumenda hic amet magni, sunt quae totam, dignissimos
      omnis nemo minima repudiandae voluptates neque atque in veritatis explicabo, eos culpa! Quidem autem iste
      veritatis laboriosam, dolores tempore rem quo cumque minima eaque possimus rerum aperiam error sed non eos totam
      vero quas. Corporis consequatur quas incidunt, rerum, aspernatur, eum ratione voluptatum molestias velit non
      impedit expedita quaerat illum similique laboriosam amet iste soluta? Sit ratione quos tempora eos, corrupti
      perferendis sint repudiandae delectus libero architecto odio cupiditate impedit? Aspernatur architecto sapiente
      fuga deleniti voluptatem, quasi, perspiciatis, officiis qui modi sequi iusto adipisci.
    </div>

    <main>
      <div class="history">
        <div class="header">
          <h1>History</h1>
          <p>Past 5 User search history</p>
        </div>
        <div class="display" v-if="history.length">
          <div v-for="(entry, index) in history" :key="index">
            <div>Passenger Class: {{ entry.pclass }} , Sex: {{ entry.sex }} , Age: {{ entry.age }} , Fare: {{ entry.fare }} , Travelled alone: {{ entry.traveled_alone }} , Embarked: {{ entry.embarked }} , Model: {{ entry.model }} , Result: {{ entry.result }}
            </div>
            <br>
          </div>
        </div>
      </div>
      <div class="form">
        <form @submit.prevent="submitForm" class="form-container">
          <div class="form-group">
            <label for="dropdown1">Passenger Class</label>
            <select id="dropdown1" v-model="dropdown1" required>
              <option disabled value="">Please select one</option>
              <option value="1">First</option>
              <option value="2">Second</option>
              <option value="3">Third</option>
            </select>
          </div>

          <div class="form-group">
            <label for="dropdown2">Sex</label>
            <select id="dropdown2" v-model="dropdown2" required>
              <option disabled value="">Please select one</option>
              <option value="0">Male</option>
              <option value="1">Female</option>
            </select>
          </div>

          <div class="form-group">
            <label for="dropdown3">Embarked</label>
            <select id="dropdown3" v-model="dropdown3" required>
              <option disabled value="">Please select one</option>
              <option value="0">Cherbourg</option>
              <option value="1">Queenstown</option>
              <option value="2">Southampton</option>
            </select>
          </div>

          <div class="form-group">
            <label for="slider1">Age (1-100):</label>
            <input type="range" id="slider1" v-model.number="slider1" min="1" max="100" required />
            <span>{{ slider1 }}</span>
          </div>

          <div class="form-group">
            <label for="slider2">Fare $ (1-520):</label>
            <input type="range" id="slider2" v-model.number="slider2" min="1" max="520" required />
            <span>{{ slider2 }}</span>
          </div>

          <div class="form-group">
            <label for="dropdown5">Travelled Alone</label>
            <select id="dropdown5" v-model="dropdown5" required>
              <option disabled value="">Please select one</option>
              <option value="1">Yes</option>
              <option value="0">No</option>
            </select>
          </div>

          <div class="foot">
            <div class="form-group">
              <label for="dropdown4">Model Selection</label>
              <select id="dropdown4" v-model="dropdown4" required>
                <option disabled value="">Please select one</option>
                <option value="logistic_regression">Logistic Regression</option>
                <option value="support_vector_machine">Support Vector Machine</option>
                <option value="k_nearest_neighbors">K-Nearest Neighbour</option>
                <option value="gaussian_naive_bayes">Guassian Naive Bayes</option>
                <option value="perceptron">Perceptron</option>
                <option value="stochastic_gradient_descent">Stochastic Gradient Descent</option>
                <option value="decision_tree">Decision Tree</option>
                <option value="random_forest">Random Forest</option>
              </select>
            </div>


            <button type="submit" class="btn">Submit</button>

          </div>

          <div v-if="result !== null">
          <h3>Result:</h3>
          <span >{{ ans }}</span>
         </div>
        </form>

        

      </div>
    </main>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      dropdown1: 1,
      dropdown2: 1,
      dropdown3: 1,
      dropdown4: 1,
      dropdown5: 1,
      slider1: 1,
      slider2: 1,
      result: null,
      ans:null,
      history:[],
      current:{}
    };
  },
  methods: {
    async submitForm() {
      try {
        const response = await axios.post(`http://127.0.0.1:8080/surv_or_not/${this.dropdown4}`, {
          "pclass": this.dropdown1,
          "sex": this.dropdown2,
          "age": this.slider1,
          "fare": this.slider2,
          "traveled_alone": this.dropdown5,
          "embarked": this.dropdown3,
          

          
        });
        this.result = response.data["survived"];
        console.log(this.result);
        if (this.result){
          this.ans="The passenger will survive."
        }
        else{
          this.ans="The passenger will not survive."
        }

        this.current= {
          "pclass": this.dropdown1,
          "sex": this.dropdown2,
          "age": this.slider1,
          "fare": this.slider2,
          "traveled_alone": this.dropdown5,
          "embarked": this.dropdown3,
          "model":this.dropdown4,
          "result": this.ans
          
        }
        this.history.unshift(this.current)
        if (this.history.length > 5) {
          this.history.pop();
        }
        
      } catch (error) {
        console.error("Error submitting form:", error);
      }
    },
  },
};
</script>

<style scoped>
* {
  padding: 0;
  margin: 0;
  outline: none;
  border: none;
  box-sizing: border-box;
  color: #fff;
}

.main {
  font-family: "Poppins", sans-serif;
  background-color: #18181b;
  padding-top: 2vh;
  color: #fff;
}

header {
  background-color: #1f2937;
  padding: 2rem;
}

nav {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
}

h1 .text {
  align-items: center;
  margin-bottom: 1.5rem;
}

.btn {
  width: 15rem;
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

.explain{
  padding: 2rem;
}


main {
  display: flex;
  flex-direction: row;
}

.history {
  display: flex;
  flex-direction: column;
  width: 40%;
  border: 2px solid white;
  border-radius: 50px;
  margin: 2rem;
}

.history .header {
  padding: 1rem;
  margin: 1rem;
  border-radius: 50px;
  background-color: rgba(0, 0, 255, 0.128);
  font-size: 1.1rem;
}

.history .display {
  padding: 1rem;
  margin: 1rem;
  border: 2px solid white;
  border-radius: 40px;
  min-height: 300px;
}

.form {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 60%;
  margin: 2rem 1.5rem 1.5rem 0.5rem;
}

.form-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  width: 100%;
  padding: 2rem;
  background-color: #1f2937;
  border: 2px solid white;
  border-radius: 50px;
}

.form-group {
  display: flex;
  flex-direction: column;
  margin-bottom: 1.7rem;
}

label {
  margin-bottom: 0.5rem;
  font-size: 1.1rem;
  font-weight: 600;
}

select,
input[type="range"] {
  padding: 1rem;
  border-radius: 25px;
  border: 1px solid #ffffff;
  background-color: #111827;
  color: #fff;
  font-size: 1rem;
  outline: none;
  transition: all 0.3s ease;
}

select:focus,
input[type="range"]:focus {
  border-color: #1b6ad3;
}

input[type="range"] {
  /* -webkit-appearance: none; */
  appearance: none;
  width: 100%;
  background: transparent;
}

input[type="range"]::-webkit-slider-runnable-track {
  width: 100%;
  height: 3px;
  background: #0ccd12;
  border-radius: 5px;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 10px;
  height: 20px;
  background: #1d4ed8;
  border-radius: 100% 100% 0% 0%;
  cursor:grab;
}

/* input[type="range"]::-moz-range-track {
  width: 100%;
  height: 10px;
  background: #1d4ed8;
  border-radius: 5px;
}

input[type="range"]::-moz-range-thumb {
  width: 20px;
  height: 20px;
  background: #1d4ed8;
  border-radius: 50%;
  cursor: pointer;
} */

input[type="range"]::-ms-track {
  width: 100%;
  height: 5px;
  background: transparent;
  border-color: transparent;
  border-width: 5px 0;
  color: transparent;
}

/* input[type="range"]::-ms-fill-lower,
input[type="range"]::-ms-fill-upper {
  background: #1d4ed8;
} */

/* input[type="range"]::-ms-thumb {
  width: 20px;
  height: 20px;
  background: #1d4ed8;
  border-radius: 50%;
  cursor: pointer;
} */

.foot {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: space-around;
}
</style>
