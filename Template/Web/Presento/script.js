const img= document.getElementById('imagecheque');

const predictionletter= document.getElementById('predictionletter');
const predictionnumber= document.getElementById('predictionnumber');





const sendmail= document.getElementById('sendmail')
sendmail.onclick=function(){
 sendemail();
}
 
const obj='{"adress": "wissem22111@gmail.com"}'
const json=JSON.parse(obj)
sendemail=async()=>{
const email = document.getElementById('email').value;
 const subject = document.getElementById('subject').value;
 const message = document.getElementById('message').value;
  const name = document.getElementById('name').value;

 console.log(email)
const url="http://127.0.0.1:8000/email/"+email+"/"+subject+"/"+message+"/"+name
console.log(url)

	await fetch(url, {
    method: "GET",
   
    
}
)}


const file= document.getElementById('fileupload');
myfunction = async()=>{var path = (window.URL || window.webkitURL).createObjectURL(file.files[0]);
   img.src=path;
   
  var formData = new FormData();
  formData.append("img", file.files[0]);


const dataToSend = formData;
let dataReceived = ""; 


const res= await fetch("http://127.0.0.1:8000/test", {
    method: "POST",
   
    //headers: { "Content-Type": "multipart/form-data" },
    body: dataToSend
})





var data = await res.json();
console.log(data)
console.log(predictionletter);
predictionletter.innerHTML=data.letter;
predictionnumber.innerHTML=data.number;
predictionnumber.setAttribute("class", "unshow");
document.getElementById("test").setAttribute("class", "unshow");
document.getElementById("oum").setAttribute("class", "unshow");
document.getElementById("ily").setAttribute("class", "unshow");
document.getElementById("predictionletter").setAttribute("class", "unshow");
console.log(`Received: ${dataReceived}`) 

}
file.addEventListener('change',function async (){
	//var tmppath = URL.createObjectURL(file.mozFullPath);
   // $("img").fadeIn("fast").attr('src',URL.createObjectURL(event.target.files[0]));
  myfunction()
  
   




})