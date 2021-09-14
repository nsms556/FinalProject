import React, {useState} from "react";

import {DislikeFilled, DislikeOutlined, LikeFilled, LikeOutlined} from "@ant-design/icons";


const Polarizing = (props) => {

    const [Like, setLike] = useState(false);
    const [Dislike, setDislike] = useState(false);

    const onClickLikes = (e) => {

        if (Dislike) {
            setDislike(!Dislike);
        }
        setLike(!Like);

        checkPolarizing(props.id, e.currentTarget.value, null);
    };

    const onClickDislikes = (e) => {

        if (Like) {
            setLike(!Like);
        }
        setDislike(!Dislike);

        checkPolarizing(props.id, null, e.currentTarget.value);
    };

    const checkPolarizing = (id, like, dislike) => {

        // Like 또는 Dislike 클릭하여 선택 시: false
        // 이미 선택된 버튼을 다시 클릭하여 취소 시: true

        props.addList(id, (like === 'false') ? 'like' : ((dislike === 'false') ? 'dislike' : null));
    };

    return (
        <>
            <button className={Like ? "btn btn-icon btn-danger" : "btn btn-icon btn-secondary"} type="button"
                    value={Like} onClick={onClickLikes}>
                    <span key="comment-basic-like">
                        {
                            Like ? (<LikeFilled/>) : (<LikeOutlined/>)
                        }
                    </span>
            </button>
            <button className={Dislike ? "btn btn-icon btn-default" : "btn btn-icon btn-secondary"} type="button"
                    value={Dislike} onClick={onClickDislikes}>
                    <span key="comment-basic-dislike">
                        {
                            Dislike ? (<DislikeFilled/>) : (<DislikeOutlined/>)
                        }
                    </span>
            </button>
        </>
    );
}

export default Polarizing;
