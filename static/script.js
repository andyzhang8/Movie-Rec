document.getElementById('movieForm').addEventListener('submit', function (e) {
    e.preventDefault();

    const movieTitle = document.getElementById('movie_title').value;
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/recommend', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `movie_title=${encodeURIComponent(movieTitle)}`,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            const referencedDiv = document.createElement('div');
            referencedDiv.classList.add('referenced-movie');
            
            const referencedTitle = document.createElement('h2');
            referencedTitle.textContent = `Your Search: ${data.referencedMovie.primaryTitle}`;
            
            const referencedDetails = document.createElement('p');
            referencedDetails.textContent = `Rating: ${data.referencedMovie.averageRating}, Year: ${data.referencedMovie.startYear}, Runtime: ${data.referencedMovie.runtimeMinutes} mins, Genres: ${data.referencedMovie.genres}`;
            
            referencedDiv.appendChild(referencedTitle);
            referencedDiv.appendChild(referencedDetails);
            resultsDiv.appendChild(referencedDiv);

            data.recommendations.forEach(movie => {
                const movieDiv = document.createElement('div');
                movieDiv.classList.add('movie');

                const title = document.createElement('h3');
                title.textContent = movie.primaryTitle;

                const rating = document.createElement('p');
                rating.textContent = `Rating: ${movie.averageRating}`;

                const similarity = document.createElement('p');
                similarity.textContent = `Similarity: ${movie.similarity.toFixed(2)}`;

                const genres = document.createElement('p');
                genres.textContent = `Genres: ${movie.genres}`;

                movieDiv.appendChild(title);
                movieDiv.appendChild(rating);
                movieDiv.appendChild(similarity);
                movieDiv.appendChild(genres);
                resultsDiv.appendChild(movieDiv);
            });
        } else {
            resultsDiv.textContent = data.message;
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        resultsDiv.textContent = 'Error fetching recommendations.';
    });
});
