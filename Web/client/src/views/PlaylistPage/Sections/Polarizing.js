import React, {useState} from "react";


const Polarizing = () => {

    const [Likes, setLikes] = useState(false);
    const [Dislikes, setDislikes] = useState(false);

    const onClickLikes = () => {
        if (Dislikes) {
            setDislikes(!Dislikes);
        }
        setLikes(!Likes);
    };

    const onClickDislikes = () => {
        if (Likes) {
            setLikes(!Likes);
        }
        setDislikes(!Dislikes);
    };

    return (
        <span>
            <button className={Likes ? "btn btn-icon btn-danger" : "btn btn-icon btn-secondary"} type="button"
                    onClick={onClickLikes}>
                <span className="btn-inner--icon"><i className="ni ni-fat-add"></i></span>
            </button>
            <button className={Dislikes ? "btn btn-icon btn-default" : "btn btn-icon btn-secondary"} type="button"
                    onClick={onClickDislikes}>
                <span className="btn-inner--icon"><i className="ni ni-fat-delete"></i></span>
            </button>
        </span>
    );
};

export default Polarizing;
