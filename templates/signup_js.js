function handleSubmit(event) {
    event.preventDefault(); 

    var firstName = document.getElementById("f_name").value;
    var lastName = document.getElementById("l_name").value;
    var email = document.getElementById("email").value;
    var password = document.getElementById("password").value;

    if (firstName === "" || lastName === "" || email === "" || password === "") {
        alert("Please fill out all fields."); 
        return false;
    }
    

    document.getElementById("success-message").style.display = "block";
    //window.location.href = "login.html"; 
    return false;
}
